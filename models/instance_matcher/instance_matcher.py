import torch
from torch import nn
from torch.nn import functional as F
import timm
from sklearn.cluster import HDBSCAN
from models.instance_matcher.vision_backbone import VisionBackbone, VisionEmbedding


class InstanceMatcher(nn.Module):
    def __init__(
            self,
            prompted_backbone,
            vision_backbone: VisionBackbone,
            device: torch.device,
    ):
        super(InstanceMatcher, self).__init__()
        self.vision_backbone = vision_backbone
        self.prompted_backbone = prompted_backbone
        self.clusterer = HDBSCAN(min_samples=1)
        self.device = device

    def forward_match(
            self,
            embeddings,
            ref_masks
    ) -> list[torch.Tensor]:
        """
        embeddings: [B, C, H, W] or [B, H, W, C]
        ref_masks: [B, H, W] boolean masks
        :returns: list of mappings [B, Num Ref Patches, 2, 2] where the last dim is the coords, and the second dim is ref
            or target. List because num of ref patches may vary.
        """
        mappings = []
        for embedding, ref_mask in zip(embeddings, ref_masks):
            # Flatten spatial dims for easier indexing: [H*W, C]
            H, W = ref_mask.shape
            flat_embedding = embedding.view(-1, embedding.shape[-1])
            flat_mask = ref_mask.view(-1)
            # Global flattened indices
            ref_indices = torch.argwhere(flat_mask).squeeze(1).to(self.device)
            rem_indices = torch.argwhere(torch.logical_not(flat_mask)).squeeze(1).to(self.device)

            # 1. Get features
            ref_feats = flat_embedding[ref_indices]  # [N_ref, C]
            remaining_feats = flat_embedding[rem_indices]  # [N_rem, C]

            # 2. Compute Similarity (Assuming cosine similarity)
            # Result shape: [N_ref, N_rem]
            similarity = torch.mm(
                F.normalize(ref_feats, dim=-1),
                F.normalize(remaining_feats, dim=-1).T
            )

            max_per_ref = torch.max(similarity, dim=1)[0]
            min_max_per_ref = torch.median(max_per_ref, dim=0)[0]
            # 3. Get best match index relative to rem_indices
            best_rel_idx = torch.argwhere(similarity > min_max_per_ref).to(self.device)

            # Map back to global flattened indices
            best_ref_global_idx = ref_indices[best_rel_idx[:, 0]]
            best_rem_global_idx = rem_indices[best_rel_idx[:, 1]]

            # 4. Convert flat indices to 2D coordinates (y, x)
            # Coordinates for Reference Points
            ref_y = best_ref_global_idx // W
            ref_x = best_ref_global_idx % W

            # Coordinates for Target Points
            target_y = best_rem_global_idx // W
            target_x = best_rem_global_idx % W

            # 5. Stack into [N_ref, 4]
            # Each entry is (x_ref, y_ref, x_target, y_target)
            maps_to = torch.stack([ref_x, ref_y, target_x, target_y], dim=1).to(self.device)

            mappings.append(maps_to)
        return mappings

    def backward_match(self, embeddings, ref_masks, matches):
        # 1. Get matches from forward_match
        # Assuming batch_backward_matches is a list of tensors [Num_matches, 4]
        # or a padded tensor [B, Num_matches, 4]
        tgt_masks = [torch.zeros_like(ref_mask).to(ref_mask.device) for ref_mask in ref_masks]
        for tgt_mask, match in zip(tgt_masks, matches):
            x_match = match[:, 2].tolist()
            y_match = match[:, 3].tolist()
            tgt_mask[y_match, x_match] = 1.
        batch_backward_matches = self.forward_match(embeddings, tgt_masks)
        batch_final_points = []

        for i, backward_matches in enumerate(batch_backward_matches):
            if len(backward_matches) == 0:
                batch_final_points.append([])
                continue

            # backward_matches: [N, 4] -> [x_m, y_m, x_br, y_br]
            # Extract the back-reference coordinates
            x_backref = backward_matches[:, 2].tolist()
            y_backref = backward_matches[:, 3].tolist()
            # 2. Direct Indexing (The "Secret Sauce")
            # Instead of 'in ref_points', we check the mask at those coordinates
            # This is an O(1) operation in C++ backend
            is_valid = torch.where(ref_masks[i, y_backref, x_backref] > 0., True, False).tolist()
            if not is_valid:
                continue
            # 3. Filter and extract target coordinates [x_match, y_match]
            x_match = backward_matches[is_valid, 0].to(self.device)
            y_match = backward_matches[is_valid, 1].to(self.device)

            batch_final_points.append(torch.stack([x_match, y_match], 1))
        return batch_final_points

    def instance_prompt_samples(
            self,
            matches
    ):
        """
        Apply density based clustering to organize the found points into instances.
        Why density based clustering?
            We do not know how many instances are there, so we should consider, that each point might be an instance or
            all belong to the same. Density based clustering solves this problem for us.

        :param matches: Matches is a batch of lists of two dimensional vectors [x_point, y_point]. Get prompts for each
            batch.
        :returns: List of prompts for each batch to run SAM2 with.
        """
        batch_prompts = []

        # matches is [Batch, Points, 2]
        for batch_item in matches:
            # 1. Convert to NumPy for HDBSCAN compatibility
            # We ensure it's on CPU and detached from the graph
            points_np = batch_item.detach().cpu().numpy()

            # 2. Initialize and fit the clustering algorithm
            # This needs to be done on CPU for now, unless we implement with cuML
            # min_samples=1 ensures even single isolated points can form a cluster
            cluster_labels = self.clusterer.fit_predict(points_np)

            # Convert labels back to torch to stay consistent with SAM2 inputs
            cluster_labels = torch.from_numpy(cluster_labels).to(batch_item.device)
            unique_labels = torch.unique(cluster_labels)

            instance_clusters = {}
            for label in unique_labels:
                # Mask points belonging to the current cluster
                cluster_points = batch_item[cluster_labels == label]
                instance_clusters[label] = cluster_points

            batch_prompts.append(instance_clusters)

        return batch_prompts

    def find_instances(
            self,
            images,
            prompts,
    ):
        """ Use prompts to segment instances with sam."""
        pass

    def filter_masks(
            self,
            ref_masks,
            pred_masks,
    ):
        """ Filter the predicted masks according to Matcher. """
        pass

    def forward(
            self,
            images,
            ref_masks,
    ):
        embeddings: VisionEmbedding = self.vision_backbone(images)
        matches = self.forward_match(
            embeddings.embedding,
            ref_masks
        )
        filtered_points = self.backward_match(
            embeddings.embedding,
            ref_masks,
            matches
        )
        prompts = self.instance_prompt_samples(
            filtered_points
        )
        pred_masks = self.find_instances(
            images,
            prompts,
        )
        return self.filter_masks(
            ref_masks,
            pred_masks
        )
