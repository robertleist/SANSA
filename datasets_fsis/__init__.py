from datasets_fsis.lvis import build as build_lvis


def build_fsis_dataset(dataset_file: str, image_set: str, args=None):
    match dataset_file:
        case 'lvis':
            return build_lvis(image_set, args)
        case 'multi':
            from datasets.transform_utils import CustomConcatDataset
            ds_list = [build_fsis_dataset(name, image_set="train", args=args) for name in args.multi_train]
            return CustomConcatDataset(
                dataset_list=ds_list,
                dataset_ratio=args.ds_weight,
                samples_per_epoch=210000
            )
        case _:
            raise ValueError(f'dataset {dataset_file} not supported')