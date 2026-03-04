r""" LVIS-92i few-shot semantic segmentation dataset """
import os
import pickle
import random

import torch
import torch.nn.functional as F
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
from datasets.transform_utils import polygons_to_bitmask
import pycocotools.mask as mask_util
from torchvision import transforms


class DatasetLVIS(Dataset):
    def __init__(
            self,
                 datapath,
            fold,
            transform,
            split,
            shot,
            use_original_imgsize,
            min_instances=2,
    ):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 10
        self.shot = shot
        self.anno_path = os.path.join(datapath, "LVIS")
        self.base_path = os.path.join(datapath, "LVIS", 'coco')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.min_instances = min_instances

        # 1. Load raw metadata
        _, self.class_ids_ori, raw_metadata = self.build_img_metadata_classwise()

        # 2. Filter categories based on min_instances requirement
        # Only keep categories that have at least one image with >= min_instances
        self.filtered_metadata = self.filter_categories(raw_metadata)

        # 3. The index now refers to the list of valid categories
        self.valid_category_ids = sorted(list(self.filtered_metadata.keys()))
        self.nclass = len(self.valid_category_ids)

    def filter_categories(self, raw_metadata):
        """ Filters the metadata to ensure every category has usable images. """
        filtered = {}
        for cat_id, images in raw_metadata.items():
            # Keep only images within this category that meet the instance count
            valid_images_for_cat = {
                img_name: data for img_name, data in images.items()
                if len(data['annotations']) >= self.min_instances
            }

            if len(valid_images_for_cat) > 0:
                filtered[cat_id] = valid_images_for_cat
        return filtered

    def __len__(self):
        # Length is now the number of unique categories available
        return len(self.valid_category_ids)

    def __getitem__(self, idx):
        # 1. Map index to a specific category
        cat_id = self.valid_category_ids[idx]

        # 2. Randomly pick an image from this category's valid pool
        cat_images = self.filtered_metadata[cat_id]
        img_name = random.choice(list(cat_images.keys()))

        # 3. Load image and ONLY instances belonging to this specific category
        image, instances, org_size = self.load_category_instances(img_name, cat_id)

        # 4. Transform and Process
        img_tensor = self.transform(image)
        processed_instances = []
        for instance in instances:
            # Standardizing mask size if necessary
            if not self.use_original_imgsize:
                # Interpolate or pad logic here if needed
                pass
            processed_instances.append(instance.permute(1, 0).float())

        return {
            'image': img_tensor,
            'instances': torch.stack(processed_instances) if processed_instances else torch.empty(0),
            'category_id': cat_id,
            'img_name': img_name,
            'img_id': int((img_name.split('.')[0]).split('/')[-1]),  # get the image id from the name
            'org_size': org_size
        }

    def load_category_instances(self, img_name, cat_id):
        img_path = os.path.join(self.base_path, img_name)
        image = Image.open(img_path).convert('RGB')
        org_size = image.size

        # Retrieve only annotations for the requested category
        annos = self.filtered_metadata[cat_id][img_name]['annotations']
        masks = [self.get_mask(anno['segmentation'], org_size) for anno in annos]

        return image, masks, org_size

    def build_img_metadata_classwise(self):
        # Standard LVIS pkl loading logic
        filename = 'lvis_train.pkl' if self.split == 'trn' else 'lvis_val.pkl'
        with open(os.path.join(self.anno_path, filename), 'rb') as f:
            anno_data = pickle.load(f)

        cat_ids = sorted(list(anno_data.keys()))
        return len(cat_ids), cat_ids, anno_data

    def build_img_metadata(self):
        # Collect all unique image names across all classes
        img_names = set()
        for cat_id in self.img_metadata_classwise:
            img_names.update(self.img_metadata_classwise[cat_id].keys())
        return sorted(list(img_names))

    def get_mask(self, segm, image_size):
        if isinstance(segm, list):
            mask = polygons_to_bitmask([np.asarray(p) for p in segm], *image_size[::-1])
        elif isinstance(segm, dict):
            mask = mask_util.decode(segm)
        else:
            mask = np.zeros(image_size[::-1])
        return torch.from_numpy(mask)

    def load_image_and_instances(self, img_name):
        # Find all annotations for this image across all categories
        img_path = os.path.join(self.base_path, img_name)
        image = Image.open(img_path).convert('RGB')
        org_size = image.size

        all_masks = []
        # Search classwise dict for this image's entries
        for cat_id in self.class_ids_ori:
            if img_name in self.img_metadata_classwise[cat_id]:
                annos = self.img_metadata_classwise[cat_id][img_name]['annotations']
                for anno in annos:
                    m = self.get_mask(anno['segmentation'], org_size)
                    all_masks.append(m)

        return image, all_masks, org_size


def build(image_set, args):
    img_size = 640
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = DatasetLVIS(
        datapath=args.data_root,
        fold=args.fold,
        transform=transform,
        shot=args.shots,  # Number of GT prompts to provide to start the sequence
        use_original_imgsize=False,
        split=image_set
    )

    return dataset