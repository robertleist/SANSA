import torch


def collate_fn(batch):
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
    return {
        'image': torch.stack(batch[0]),
        'instances': batch[1], # List of tensors
        'org_size': batch[2],
        'image_name': batch[3],
        'image_id': batch[4],
        'category_id': batch[4],
    }
