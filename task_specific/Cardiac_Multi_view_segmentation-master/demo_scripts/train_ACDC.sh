# configs/semi_supervised_learning/ACDC/supervised/baseline.json
# taskset -c 4,5,6,7 python train.py --json_config_path 'configs/ACDC/supervised/baseline.json' --log --intensity_norm_type z_score  --gpu 1
taskset -c 0,1,2,3 python train.py --json_config_path 'configs/ACDC/supervised/adv_bias.json' --log --intensity_norm_type z_score  --gpu 1 --adv_training
# taskset -c 8,9,10,11 python train.py --json_config_path 'configs/ACDC/supervised/adv_bias_ce.json' --log --intensity_norm_type z_score  --gpu 0 --adv_training
taskset -c 0,1,2,3 python train.py --json_config_path 'configs/ACDC/supervised/adv_bias.json' --log --intensity_norm_type z_score  --gpu 1 --adv_training
