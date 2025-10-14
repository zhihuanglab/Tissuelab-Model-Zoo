# Created by cc215 at 02/05/19
# Modified by cc215 at 11/12/19

# This code is for training basic segmentation networks (Unet-64)
# Scenario: learn to segment cardiac short-axis images
# Steps:
#  1. define the segmentation network and optimiser
#  2. fetch images tuples from the disk to train the segmentation
#  3. calculate standard cross entropy loss
#  4. (optional), if adv_training=True, perform adversarial data augmentation and calculate adv regularization loss
#  5. optimize the network, back to step 2.
from __future__ import print_function
import argparse
import os
from common_utils.io import check_dir
from os.path import exists, join
import gc
import socket
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from model.base_segmentation_model import SegmentationModel
from model.model_utils import makeVariable
from common_utils.metrics import print_metric
from common_utils.load_args import Params
from common_utils.basic_operations import intensity_norm_fn, construct_input
from get_adv_augmentor import get_default_augmentor
from dataset_loader.dataset_picker import get_train_eval_datsets


def train_network(experiment_name: str,
                  dataset: list,
                  segmentor_opt: dict,
                  experiment_opt: dict,
                  save_dir: str,
                  ):
    '''

    :param experiment_name:
    :param dataset:
    :return:
    '''
    # output setting
    global training_opt, adversarial_training, log, intensity_norm_type
    # ========================Define models==================================================#
    optimizer_name = 'adam'
    try:
        optimizer_name = segmentor_opt['optimizer_name']
    except:
        pass
    print('use  optimizer: {} with. lr {}'.format(
        optimizer_name, segmentor_opt["lr"]))
    segmentation_model = SegmentationModel(network_type=segmentor_opt["network_type"], num_classes=segmentor_opt["num_classes"],
                                           resume_path=segmentor_opt["resume_path"],
                                           optimizer_name=optimizer_name,
                                           decoder_dropout=segmentor_opt["decoder_dropout"],
                                           use_gpu=segmentor_opt["use_gpu"], lr=segmentor_opt["lr"]
                                           )

    # =========================dataset config==================================================#
    train_set = dataset[0]
    validate_set = dataset[1]
    batch_size = segmentor_opt["batch_size"]
    keep_origin_data = True

    batch_size /= 2

    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=int(
        batch_size), shuffle=True, drop_last=True, pin_memory=False)
    validate_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=int(batch_size), shuffle=True, pin_memory=False,
                                 drop_last=False)
    best_score = -10000
    # =========================<<<<<start logging>>>>>>>>=============================>

    if log:
        machine_name = socket.gethostname().split('.')[0]
        log_dir = './runs/'
        check_dir(log_dir, create=True)
        writer = SummaryWriter(log_dir=log_dir+experiment_name +
                               '.'+machine_name, comment=experiment_name, purge_step=0)

    # =========================<<<<<start training>>>>>>>>=============================>
    i_iter = 0
    stop_flag = False
    for i_epoch in range(segmentor_opt['n_epochs']):
        # initialize loss dict
        gc.collect()  # collect garbage
        g_count = 0
        total_loss = 0.
        device = torch.device(
            'cuda') if segmentor_opt["use_gpu"] else torch.device('cpu')
        loss_dict = {
            'train/loss': torch.tensor(0., device=device),
        }

        # train
        for b_iter, labelled_batch in enumerate(train_loader):
            if stop_flag:
                break
            gc.collect()  # collect garbage
            torch.cuda.empty_cache()

            # step 1: forward a batch of labelled images
            segmentation_model.train()
            segmentation_model.reset_optimizers()
            image, gt = labelled_batch['image'], labelled_batch['label']
            if keep_origin_data:
                image_orig, gt_orig = labelled_batch['origin_image'], labelled_batch['origin_label']
                image = torch.cat([image, image_orig], dim=0)
                gt = torch.cat([gt, gt_orig], dim=0)

            # step 2-4: supervised learning w/ or w/o adversarial data augmentation
            supervised_segmentation_loss = supervised_learning(input_image=image, input_gt=gt, segmentation_model=segmentation_model,
                                                               use_gpu=segmentor_opt["use_gpu"], adversarial_training=adversarial_training)

            if torch.isnan(supervised_segmentation_loss):
                print('NAN detected')
                continue
            print('{} : {} loss {:.3f}'.format(str(i_epoch), str(
                b_iter), supervised_segmentation_loss.item()))
            total_loss += supervised_segmentation_loss.item()
            loss_dict['train/loss'] += supervised_segmentation_loss.item()
            supervised_segmentation_loss.backward()
            segmentation_model.optimize_params()
            segmentation_model.reset_loss()

            if i_iter > segmentor_opt["max_iteration"]:
                stop_flag = True

            if log and (i_iter == 0 or i_iter % 100 == 0):
                print('logging w. tensorboard')
                for loss_name, loss_value in loss_dict.items():
                    writer.add_scalar(
                        loss_name, (loss_value/(b_iter+1)), i_iter)
            g_count += 1
            i_iter += 1
            torch.cuda.empty_cache()
            if b_iter == 200:
                # evaluate and save model checkpoint every 200 iters
                break

        if stop_flag:
            break
        print('{} network: {} epoch {} training loss iter: {}, total  loss: {}'.
              format(experiment_name, segmentor_opt["network_type"], i_epoch, g_count, str(total_loss / (1.0 * g_count))))

        #
        # =========================<<<<<start evaluating>>>>>>>>=============================>
        segmentation_model.running_metric.reset()
        segmentation_model.eval()
        for b_iter, batch in enumerate(validate_loader):
            random_sax_image, random_sax_gt = batch['image'], batch['label']
            if keep_origin_data:
                image_orig, gt_orig = labelled_batch['origin_image'], labelled_batch['origin_label']
                random_sax_image = torch.cat(
                    [random_sax_image, image_orig], dim=0)
                random_sax_gt = torch.cat([random_sax_gt, gt_orig], dim=0)
            random_sax_image = intensity_norm_fn(
                intensity_norm_type)(random_sax_image)
            random_sax_image_V = makeVariable(
                random_sax_image, type='float', use_gpu=segmentor_opt["use_gpu"], requires_grad=True)
            segmentation_model.evaluate(input=random_sax_image_V,
                                        targets_npy=random_sax_gt.numpy())
            # faster training when validation set is too large.
            # if b_iter>200:
            #     break

        score = print_metric(segmentation_model.running_metric, name='')
        # keep the best model
        curr_score = score['Mean IoU : \t']
        curr_acc = score['Mean Acc : \t']
        print('val IOU', curr_score)
        if log:
            writer.add_scalar('iou/validate_iou', curr_score, i_epoch)
            writer.add_scalar('acc/validate_acc', curr_acc, i_epoch)

        if best_score < curr_score:
            best_score = curr_score
            segmentation_model.save_model(
                save_dir, epoch_iter='best', model_prefix=segmentor_opt["network_type"])
            segmentation_model.save_testing_images_results(
                save_dir, epoch_iter='best', max_slices=5)

        ###########save outputs ####################################################################
        if i_epoch % experiment_opt["output"]["save_epoch_every_num_epochs"] == 0 or i_epoch == 1:
            segmentation_model.save_model(
                save_dir, epoch_iter=i_epoch, model_prefix=segmentor_opt["network_type"])
            segmentation_model.save_testing_images_results(
                save_dir, epoch_iter=i_epoch, max_slices=5)
            gc.collect()  # collect garbage

        if segmentation_model.scheduler is not None:
            segmentation_model.scheduler.step()


def supervised_learning(input_image, input_gt, segmentation_model, adversarial_training=False, use_gpu: bool = True):
    '''
    supervised learning with the given labelled data, loss function by default all use cross entropy.
    :param input_image: 4d image tensor
    :param input_gt: 3D label tensor
    :param segmentation_model:the model for prediction
    :param adversarial_training: bool: if true, perform adversarial data augmentation
    :param use_gpu: bool: if use gpu
    :return:
    loss for backpropagation
    '''

    '''
    standard supervised learning using cross entropy
    '''
    global experiment_opt, intensity_norm_type, debug
    assert not input_gt is None, 'standard supervised learning requires ground truth'
    input_image = intensity_norm_fn(intensity_norm_type)(input_image)
    image_V = makeVariable(input_image.detach().clone(),
                           type='float', use_gpu=use_gpu, requires_grad=False)
    gt_V = makeVariable(input_gt.detach().clone(), type='long',
                        use_gpu=use_gpu, requires_grad=False)

    augmentor_opt = None
    torch.cuda.empty_cache()
    divergence_types = ['mse', 'contour']
    divergence_weights = [1.0, 0.5]
    n_iter = 1
    try:
        augmentor_opt = experiment_opt["adversarial_augmentation"]
        policy_name = augmentor_opt['policy_name']
    except:
        print('use default transformation config')
        policy_name = 'advchain'
        pass

    l = torch.tensor(0., device=image_V.device)

    segmentation_model.train()
    segmentation_model.model.zero_grad()
    y_pred = segmentation_model.forward(image_V)
    l = segmentation_model.get_loss(pred=y_pred, targets=gt_V)
    # onehot_gt = construct_input(segmentation=gt_V,num_classes=y_pred.size(1),apply_softmax=False, is_labelmap=True,use_gpu=True)

    if (not augmentor_opt is None) and adversarial_training and (not torch.isnan(l)):
        try:
            divergence_types = augmentor_opt["divergence_types"]
            divergence_weights = augmentor_opt["divergence_weights"]
            n_iter = augmentor_opt["n_iter"]
        except:
            print('use default transformation config')

        augmentor = get_default_augmentor(
            policy_name=policy_name,
            data_size=[*image_V.size()],
            divergence_types=divergence_types,
            divergence_weights=divergence_weights,
            debug=False,
            use_gpu=True,
        )

        supervised_consistency_loss = augmentor.adversarial_training(data=image_V, init_output=None,
                                                                     model=segmentation_model.model, lazy_load=False, n_iter=n_iter,
                                                                     )
        print('D_l:{:.4f}'.format(supervised_consistency_loss.item()))
        l += supervised_consistency_loss
    torch.cuda.empty_cache()
    return l


# ========================= config==================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cardiac segmentation network training')
    parser.add_argument("--json_config_path", type=str, default='configs/composite_train.json',
                        help='path of configurations')
    parser.add_argument("--adv_training", action='store_true', default=False,
                        help='enable adversarial training')
    parser.add_argument("--save_dir", type=str,
                        default="./result/",
                        help='path to resume the models')
    parser.add_argument('--gpu', default=0,
                        help='select GPU by masking shell environment variable CUDA_VISIBLE_DEVICES')
    parser.add_argument('--intensity_norm_type', default='z_score', type=str,
                        help="'min_max': perform 0-1 rescale; 'z_score': use z_score intensity normalization")
    parser.add_argument("--log", action='store_true', default=False,
                        help='use tensorboard to track training progress')
    parser.add_argument("--debug", action='store_true', default=False,
                        help='debugging')
    # ========================= initialize training settings==================================================#
    # first load basic settings and then load args, finally load experiment configs
    # enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    cudnn.benchmark = True

    training_opt = parser.parse_args()
    if exists(training_opt.json_config_path):
        print('load params from {}'.format(training_opt.json_config_path))
        experiment_opt = Params(training_opt.json_config_path).dict
    else:  #
        raise FileNotFoundError
    os.environ["CUDA_VISIBLE_DEVICES"] = str(training_opt.gpu)
    # input dataset setting
    data_opt = experiment_opt['data']
    intensity_norm_type = training_opt.intensity_norm_type
    datasets = get_train_eval_datsets(
        data_opt=data_opt, dataset_config_name='CardiacUKBBDataset')

    # ========================= start training ==================================================#

    debug = training_opt.debug
    log = training_opt.log
    adversarial_training = training_opt.adv_training
    print('enable adv training:', adversarial_training)

    experiment_name = (training_opt.json_config_path.split('.')[
                       0]).replace('configs/', '')
    print('exp name:', experiment_name)
    save_dir = join(training_opt.save_dir, experiment_name)
    print('save dir:', save_dir)
    check_dir(save_dir, create=True)

    train_network(experiment_name=experiment_name,
                  dataset=datasets,
                  segmentor_opt=experiment_opt['segmentation_model'],
                  experiment_opt=experiment_opt,
                  save_dir=save_dir,
                  )
