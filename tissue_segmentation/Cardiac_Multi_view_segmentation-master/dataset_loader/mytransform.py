# Created by cc215 at 05/02/20
# Enter feature description here
# Enter scenario name here
# Enter steps here

import torchsample.transforms as ts
from dataset_loader._utils.affine_transform import MyRandomFlip, MySpecialCrop
from dataset_loader._utils.elastic_transform import MyElasticTransform, MyElasticTransformCoarseGrid
from dataset_loader._utils.intensity_transform import MyNormalizeMedicPercentile, RandomBrightnessFluctuation,MyNormalizeMedic
class Transformations:
    def __init__(self, data_aug_policy_name, pad_size=(80, 80, 1), crop_size=(80, 80, 1)):
        self.name = data_aug_policy_name
        self.pad_size = pad_size
        self.crop_size = crop_size

    def get_transformation(self):

        aug_congig_transform = {
            'no_aug':self.no_aug,
            'UKBB_affine_elastic': self.UKBB_affine_elastic,
            'UKBB_affine_aug': self.UKBB_affine_aug,
            'UKBB_advanced': self.UKBB_advanced,
            'UKBB_advancedv2': self.UKBB_advanced_v2,
            'UKBB_advancedv3': self.UKBB_advanced_v3,
            'UKBB_advanced_z_score': self.UKBB_advanced_z_score,
            'UKBB_advancedv4': self.UKBB_advanced_v4,


        }[self.name]()

        return aug_congig_transform[1]

    def get_transform(self, config):
        train_transform = ts.Compose([ts.PadNumpy(size=self.pad_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                       ## intensity  transform
                                      RandomBrightnessFluctuation(p=config['intensity_prob'],flag=[True, False]),

                                      ## geometric transformation
                                      MyRandomFlip(h=config['flip_flag'][0], v=config['flip_flag'][1],
                                                   p=config['flip_flag'][2]),
                                      MyElasticTransform(is_labelmap=[False, True], p_thresh=config['elastic_prob']),
                                      ts.RandomAffine(rotation_range=config['rotate_val'],
                                                      translation_range=config['shift_val'],
                                                      shear_range=config['shear_val'],
                                                      zoom_range=config['scale_val'], interp=('bilinear', 'nearest')),

                                      MySpecialCrop(size=self.crop_size, crop_type=0),
                                      MyNormalizeMedic(norm_flag=(True, False)),
                                      ts.TypeCast(['float', 'long'])
                                      ])

        valid_transform = ts.Compose([ts.PadNumpy(size=self.pad_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      MySpecialCrop(size=self.crop_size, crop_type=0),
                                      MyNormalizeMedic(norm_flag=(True, False)),
                                      ts.TypeCast(['float', 'long'])

                                      ])
        aug_valid_transform = train_transform

        return {'train': train_transform, 'validate': valid_transform,
                'aug_validate': aug_valid_transform}

    
    
    def get_transform_v2(self, config):
        train_transform = ts.Compose([ts.PadNumpy(size=self.pad_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      
                                 
                                      ## geometric transformation
                                      MyRandomFlip(h=config['flip_flag'][0], v=config['flip_flag'][1],
                                                   p=config['flip_flag'][2]),
                                                        ## intensity  transform
                                      RandomBrightnessFluctuation(p=config['intensity_prob'],flag=[True, False]),

                                      MyElasticTransform(is_labelmap=[False, True], p_thresh=config['elastic_prob']),
                                      ts.RandomAffine(rotation_range=config['rotate_val'],
                                        translation_range=config['shift_val'],
                                        shear_range=config['shear_val'],
                                        zoom_range=config['scale_val'], interp=('bilinear', 'nearest')),

                                      MySpecialCrop(size=self.crop_size, crop_type=0),
                                      MyNormalizeMedicPercentile(norm_flag=(True, False),min_val=0,max_val=1,perc_threshold=(0,100)),
                                      ts.TypeCast(['float', 'long'])
                                      ])

        valid_transform = ts.Compose([ts.PadNumpy(size=self.pad_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      MySpecialCrop(size=self.crop_size, crop_type=0),
                                      MyNormalizeMedicPercentile(norm_flag=(True, False),min_val=0,max_val=1,perc_threshold=(0,100)),                                      
                                      ts.TypeCast(['float', 'long'])
                                      ])
        aug_valid_transform = train_transform

        return {'train': train_transform, 'validate': valid_transform,
                'aug_validate': aug_valid_transform}


    def get_transform_v3(self, config):
        train_transform = ts.Compose([ts.PadNumpy(size=self.pad_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      
                                 
                                      ## geometric transformation
                                      MyRandomFlip(h=config['flip_flag'][0], v=config['flip_flag'][1],
                                                   p=config['flip_flag'][2]),
                                                        ## intensity  transform

                                      MyElasticTransform(is_labelmap=[False, True], p_thresh=config['elastic_prob']),
                                      ts.RandomAffine(rotation_range=config['rotate_val'],
                                        translation_range=config['shift_val'],
                                        shear_range=config['shear_val'],
                                        zoom_range=config['scale_val'], interp=('bilinear', 'nearest')),
                                      RandomBrightnessFluctuation(p=config['intensity_prob'],flag=[True, False]),
                                      ts.RandomCrop(size=self.crop_size),
                                      MyNormalizeMedicPercentile(norm_flag=(True, False),min_val=0,max_val=1,perc_threshold=(1,99)),
                                      ts.TypeCast(['float', 'long'])
                                      ])

        valid_transform = ts.Compose([ts.PadNumpy(size=self.pad_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      MySpecialCrop(size=self.crop_size, crop_type=0),
                                      MyNormalizeMedicPercentile(norm_flag=(True, False),min_val=0,max_val=1,perc_threshold=(1,99)),                                      
                                      ts.TypeCast(['float', 'long'])
                                      ])
        aug_valid_transform = ts.Compose([ts.PadNumpy(size=self.pad_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      ## geometric transformation
                                      MyRandomFlip(h=config['flip_flag'][0], v=config['flip_flag'][1],
                                                   p=config['flip_flag'][2]),
                                                        ## intensity  transform
                                      RandomBrightnessFluctuation(p=config['intensity_prob'],flag=[True, False]),

                                      MyElasticTransform(is_labelmap=[False, True], p_thresh=config['elastic_prob']),
                                      ts.RandomAffine(rotation_range=config['rotate_val'],
                                        translation_range=config['shift_val'],
                                        shear_range=config['shear_val'],
                                        zoom_range=config['scale_val'], interp=('bilinear', 'nearest')),

                                      MyNormalizeMedicPercentile(norm_flag=(True, False),min_val=0,max_val=1,perc_threshold=(1,99)),
                                      MySpecialCrop(size=self.crop_size,crop_type=0),
                                      ts.TypeCast(['float', 'long'])
                                      ])

        return {'train': train_transform, 'validate': valid_transform,
                'aug_validate': aug_valid_transform}

    
    
    def get_transform_v4(self, config):
        train_transform  = ts.Compose([ts.PadNumpy(size=self.pad_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      ## geometric transformation
                                      MyRandomFlip(h=config['flip_flag'][0], v=config['flip_flag'][1],
                                                   p=config['flip_flag'][2]),
                                                        ## intensity  transform
                                      RandomBrightnessFluctuation(p=config['intensity_prob'],flag=[True, False]),

                                      MyElasticTransform(is_labelmap=[False, True], p_thresh=config['elastic_prob']),
                                      ts.RandomAffine(rotation_range=config['rotate_val'],
                                        translation_range=config['shift_val'],
                                        shear_range=config['shear_val'],
                                        zoom_range=config['scale_val'], interp=('bilinear', 'nearest')),

                                      ts.RandomCrop(size=self.crop_size),
                                      MyNormalizeMedic(norm_flag=(True, False)),
                                      ts.TypeCast(['float', 'long'])
                                      ])

        valid_transform = ts.Compose([ts.PadNumpy(size=self.pad_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      MySpecialCrop(size=self.crop_size, crop_type=0),
                                      MyNormalizeMedic(norm_flag=(True, False)),
                                      ts.TypeCast(['float', 'long'])
                                      ])
        aug_valid_transform = ts.Compose([ts.PadNumpy(size=self.pad_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      ## geometric transformation
                                      MyRandomFlip(h=config['flip_flag'][0], v=config['flip_flag'][1],
                                                   p=config['flip_flag'][2]),
                                                        ## intensity  transform
                                      RandomBrightnessFluctuation(p=config['intensity_prob'],flag=[True, False]),

                                      MyElasticTransform(is_labelmap=[False, True], p_thresh=config['elastic_prob']),
                                      ts.RandomAffine(rotation_range=config['rotate_val'],
                                        translation_range=config['shift_val'],
                                        shear_range=config['shear_val'],
                                        zoom_range=config['scale_val'], interp=('bilinear', 'nearest')),

                                      MySpecialCrop(size=self.crop_size,crop_type=0),
                                      MyNormalizeMedic(norm_flag=(True, False)),
                                      ts.TypeCast(['float', 'long'])
                                      ])

        return {'train': train_transform, 'validate': valid_transform,
                'aug_validate': aug_valid_transform}
    
    
    def no_aug(self):
        config = {
            ## affine augmentation
            'flip_flag': [False, False, 0.0],
            'shift_val': (0., 0.),
            'rotate_val': 0,
            'scale_val': (1., 1.),
            'rotate_groups': [],
            ## deformation aug
            'elastic_prob': 0.,
            'shear_val': 0,
            'intensity_prob':0,

        }
        transform = self.get_transform(config)
        return config,transform


    def UKBB_affine_aug(self):
        config,transform =  self.no_aug()
        config['flip_flag'] = [True, True, 0.2]
        config['shift_val'] = (0.1, 0.1)
        config['rotate_val'] = 180
        config['scale_val'] = (0.7, 1.4)
        transform = self.get_transform(config)
        return config,transform



    def UKBB_affine_elastic(self):
        config,transform = self.UKBB_affine_aug()
        config['elastic_prob'] = 0.5
        transform = self.get_transform(config)
        return config,transform


    def UKBB_advanced(self):
        config,transform = self.UKBB_affine_aug()
        config['elastic_prob'] = 0.5
        config['intensity_prob'] = 0.5
        transform = self.get_transform(config)
        return config,transform


    def UKBB_advanced_v2(self):
        config,transform = self.UKBB_affine_aug()
        config['flip_flag'] = [True, True, 0.5]
        config['shift_val'] = (0.1, 0.1)
        config['rotate_val'] = 30
        config['elastic_prob'] = 0.5
        config['intensity_prob'] = 0.5
        transform = self.get_transform_v2(config)
        return config,transform


    def UKBB_advanced_v3(self):
        config,transform = self.UKBB_affine_aug()
        config['flip_flag'] = [True, True, 0.5]
        config['shift_val'] = (0, 0.)
        config['rotate_val'] = 30
        config['elastic_prob'] = 0.5
        config['intensity_prob'] = 0.5
        transform = self.get_transform_v3(config)
        return config,transform


    def UKBB_advanced_z_score(self):
        ## with z_score intensity normalization
        config,transform = self.UKBB_affine_aug()
        config['flip_flag'] = [True, True, 0.5]
        config['shift_val'] = (0, 0.)
        config['rotate_val'] = 30
        config['elastic_prob'] = 0.5
        config['intensity_prob'] = 0.5
        transform = self.get_transform_v4(config)
        return config,transform
    
    def UKBB_advanced_v4(self):
        ## with 180 degree
        config,transform = self.UKBB_affine_aug()
        config['flip_flag'] = [True, True, 0.5]
        config['shift_val'] = (0, 0.)
        
        config['rotate_val'] = 180
        config['elastic_prob'] = 0.5
        config['intensity_prob'] = 0.5
        transform = self.get_transform_v3(config)
        return config,transform

