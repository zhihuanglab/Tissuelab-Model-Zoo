from dataset_loader.cardiac_UKBB_dataset import CardiacUKBBDataset
from dataset_loader.mytransform import Transformations  #

def get_train_eval_datsets(data_opt,dataset_config_name="CardiacUKBBDataset"):
    """[summary]
    get train and validate datasets from disk
    Args:
        data_opt ([dict]): [a dict with detailed configurations]
        dataset ([the class name of dataset]):CardiacUKBBDataset
    Returns:
        train and validation sets [list]: [train_set,val_set]
    """
    data_aug_policy_name = data_opt["data_aug_policy"]
    tr = Transformations(data_aug_policy_name=data_opt["data_aug_policy"], pad_size=data_opt['pad_size'],
                         crop_size=data_opt['crop_size']).get_transformation()
    
    if dataset_config_name =='CardiacUKBBDataset':
        train_set = CardiacUKBBDataset(root_dir=data_opt["train_dir"], num_classes=data_opt["num_classes"],
                                    readable_frames=data_opt["readable_frames"],
                                    if_resample=data_opt["if_resample"],
                                    new_spacing=data_opt["new_spacing"],
                                    keep_z_spacing=data_opt["keep_z_spacing"],
                                    image_format_name=data_opt["image_format_name"],
                                    label_format_name=data_opt["label_format_name"],
                                    transform=tr['train'],
                                    no_aug_transform=tr['validate'],
                                    use_cache=data_opt['use_cache'],
                                    myocardium_seg=data_opt['myocardium_only'],
                                    ignore_black_slices=data_opt["ignore_black_slices"],
                                        keep_orig_image_label_pair=True
                                    )

        validate_set = CardiacUKBBDataset(root_dir=data_opt["validate_dir"], num_classes=data_opt["num_classes"],
                                        image_format_name=data_opt["image_format_name"],
                                        label_format_name=data_opt["label_format_name"],
                                        transform=tr['aug_validate'],
                                        no_aug_transform=tr['validate'],
                                        readable_frames=data_opt["readable_frames"],
                                        if_resample=data_opt["if_resample"],
                                        new_spacing=data_opt["new_spacing"],
                                        keep_z_spacing=data_opt["keep_z_spacing"],
                                        use_cache=data_opt['use_cache'],
                                        myocardium_seg=data_opt['myocardium_only'],
                                        ignore_black_slices=False,
                                        keep_orig_image_label_pair=True

                                        )
    

    else: 
        raise NotImplementedError

    datasets = [train_set, validate_set]
    print('train_{}_with_{}_datasize:{}'.format(data_opt['dataset_name'], data_aug_policy_name,str(train_set.datasize)))
    print('validate_{}_with_{}_datasize:{}'.format(data_opt['dataset_name'], data_aug_policy_name,str(validate_set.datasize)))

    return datasets
