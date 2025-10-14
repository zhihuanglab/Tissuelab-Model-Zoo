## train Unet 64 with SGD w.o  adv training
## lr =0.001
CUDA_VISISBLE_DEVICES=4 taskset -c 16,17,18,19 python train.py --json_config_path 'configs/baseline_SGD.json' --log
python train.py --json_config_path 'configs/composite_train_SGD.json' --log --adv_training

## lr =0.00001, finetuning (lory)
CUDA_VISISBLE_DEVICES=3 taskset -c 12,13,14,15 python train.py --json_config_path 'configs/baseline_Adam_finetune.json' --log
## lr =0.00001, finetuning (pudsey,)
python train.py --json_config_path 'configs/composite_train_Adam_finetune.json' --log --adv_training

python train.py --json_config_path 'configs/bias_train_SGD.json' --log --adv_training

## lr =0.00001, finetuning (pudsey, independently optimize)
python train.py --json_config_path 'configs/composite_independent_train_SGD.json' --log --adv_training --gpu 1

## lr =0.00001, finetuning z_score
python train.py --json_config_path 'configs/baseline_SGD_z_score.json' --log --gpu 1

python train.py --json_config_path 'configs/baseline_Adam_z_score.json' --log --gpu 0
## lr =0.00001, finetuning z_score
python train.py --json_config_path 'configs/baseline_Adam_z_score_IN_UNet_64.json' --log --gpu 0
## lr =0.00001, finetuning min max with 1-99 percentile
python train.py --json_config_path 'configs/composite_train_Adam_finetune_aug_v3_IN_UNet_64.json' --log --gpu 1 --adv_training

## lr =0.00001, finetuning z_score
python train.py --json_config_path 'configs/baseline_Adam_v3_IN_UNet_64.json' --log --gpu 0

## add intensity norm to the adv data augmentor
taskset -c 4,5,6,7 python train.py --json_config_path 'configs/baseline_Adam_finetune_v4_50.json' --log --intensity_norm_type z_score  --gpu 1
taskset -c 8,9,10,11 python train.py --json_config_path 'configs/baseline_Adam_finetune_v4_bias_50.json' --log --intensity_norm_type 'z_score' --adv_training --gpu 2

taskset -c 12,13,14,15 python train.py --json_config_path 'configs/baseline_Adam_finetune_v4_composite_50.json' --log --intensity_norm_type z_score  --adv_training  --gpu 3
taskset -c 32,33,34,35 python train.py --json_config_path 'configs/baseline_Adam_finetune_v4_composite_50_kl.json' --log --intensity_norm_type z_score  --adv_training  --gpu 8


## compare random data aug vs adv data aug
configs/baseline_Adam_finetune_v4_composite_50_independent_mse_random.json
taskset -c 0,1,2,3 python train.py --json_config_path 'configs/baseline_Adam_finetune_v4_composite_50_chain_mse_random.json' --log --intensity_norm_type z_score  --adv_training  --gpu 0
taskset -c 4,5,6,7 python train.py --json_config_path 'configs/baseline_Adam_finetune_v4_composite_50_chain_mse_adv.json' --log --intensity_norm_type z_score  --adv_training  --gpu 1
## monal05 machine, run without power iteration.
taskset -c 36,37,38,39 python train.py --json_config_path 'configs/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_no_power.json' --log --intensity_norm_type z_score  --adv_training  --gpu 9
## lory, machine, run with random select adv compose.
## lory, machine, run with random select adv compose without power iteration.
taskset -c 24,25,26,27 python train.py --json_config_path 'configs/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_random_select_no_power.json' --log --intensity_norm_type z_score  --adv_training  --gpu 6
## lory, machine, run with random select rand compose 
taskset -c 24,25,26,27 python train.py --json_config_path 'configs/baseline_Adam_finetune_v4_composite_50_chain_mse_random_random_select.json' --log --intensity_norm_type z_score  --adv_training  --gpu 6

taskset -c 12,13,14,15 python train.py --json_config_path 'configs/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_no_power.json' --log --intensity_norm_type z_score  --adv_training  --gpu 4


taskset -c 12,13,14,15 python train.py --json_config_path 'configs/baseline_Adam_finetune_v4_composite_50_bias_mse_adv_no_power.json' --log --intensity_norm_type z_score  --adv_training  --gpu 4

