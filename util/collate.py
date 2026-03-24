import torch


def collate_fn(batch):
    has_cellpose = 'cellpose_target' in batch[0]
    if has_cellpose:
        batch = list(
            zip(
                *[(
                    item['image'],
                    item['instances'],
                    item['org_size'],
                    item['img_name'],
                    item['img_id'],
                    item['category_id'],
                    item['cellpose_target']
                ) for item in batch]
            )
        )
    else:
        batch = list(
            zip(
                *[(
                    item['image'],
                    item['instances'],
                    item['org_size'],
                    item['img_name'],
                    item['img_id'],
                    item['category_id']
                ) for item in batch]
            )
        )

    out = {
        'image': torch.stack(batch[0]),
        'instances': batch[1], # List of tensors
        'org_size': batch[2],
        'image_name': batch[3],
        'image_id': batch[4],
        'category_id': batch[5],
    }

    if has_cellpose:
        out['cellpose_target'] = torch.stack(batch[6])

    return out
