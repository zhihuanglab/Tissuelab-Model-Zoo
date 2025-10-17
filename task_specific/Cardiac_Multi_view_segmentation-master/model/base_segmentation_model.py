# Created by cc215 at 02/05/19
# segmentation model definition goes here

import os
from os.path import join
import torch.nn as nn
import torch
import torch.optim as optim
import gc

from model.init_weight import init_weights
from model.unet import UNet, INUnet
from model.model_utils import makeVariable
from common_utils.loss import cross_entropy_2D
from common_utils.metrics import runningScore
from common_utils.save import save_list_results_as_png
from model.model_utils import get_scheduler, ExponentialMovingAverage


class SegmentationModel(nn.Module):
    def __init__(self, network_type, in_channels=1, num_classes=2,
                 decoder_dropout=None, use_gpu=True, lr=0.001,
                 resume_path=None, optimizer_name='adam',
                 use_ema=False
                 ):
        '''
        :param network_type: string
        :param num_domains: int
        :param in_channels: int
        :param num_classes: int
        :param encoder_dropout: float
        :param decoder_dropout: float
        :param use_gpu: bool
        :param lr: float
        :param resume_path: string
        :param optimizer_name: string, specify the name of optimizers. 
        :param use_ema: boolean: when true, it will apply exponetial moving average to update model parameters.
        '''
        super(SegmentationModel, self).__init__()
        self.network_type = network_type
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.lr = lr
        self.in_channels = in_channels
        self.decoder_dropout = decoder_dropout if isinstance(
            decoder_dropout, float) else None

        self.model = self.get_network_from_model_library(self.network_type)
        # print number of paramters
        self.resume_path = resume_path
        self.init_model(network_type)
        if self.use_gpu:
            self.model.cuda()
        self.scheduler = None
        self.use_ema = use_ema

        self.set_optmizers(optimizer_name)
        self.running_metric = self.set_running_metric()  # cal iou score during training

        self.cur_eval_images = None
        self.cur_eval_predicts = None
        self.cur_eval_gts = None  # N*H*W

        self.loss = 0.

    def get_network_from_model_library(self, network_type):
        model = None
        model_candidates = ['UNet_64', 'IN_UNet_64', 'UNet_16']
        assert network_type in model_candidates, 'currently, we only support network types: {}, but found {}'.format(
            str(model_candidates), network_type)

        if network_type == 'UNet_64' or network_type == 'UNet_16':
            feature_scale = 4 if network_type == 'UNet_16' else 1
            model = UNet(input_channel=self.in_channels, num_classes=self.num_classes, feature_scale=feature_scale,
                         norm=nn.BatchNorm2d,
                         dropout=self.decoder_dropout)
            print('init {}'.format(network_type))
        elif network_type == 'IN_UNet_64':
            # Unet with instance normalization.
            model = INUnet(input_channel=self.in_channels, num_classes=self.num_classes, feature_scale=1,
                           norm=nn.BatchNorm2d,
                           dropout=self.decoder_dropout)
            print('init {}'.format(network_type))
        else:
            raise NotImplementedError

        return model

    def init_model(self, network_type):
        resume_path = self.resume_path
        init_weights(self.model, init_type='kaiming')
        # print('init ', network_type)
        if not resume_path is None:
            if not resume_path == '':
                assert os.path.exists(
                    resume_path), 'path: {} must exist'.format(resume_path)
                
                # Set map_location based on use_gpu
                map_location = None if self.use_gpu else torch.device('cpu')
                
                if '.pkl' in resume_path:
                    try:
                        self.model.load_state_dict(torch.load(resume_path, map_location=map_location)[
                                                   'model_state'], strict=True)
                    except:
                        print('fail to load, loose the constraint')
                        self.model.load_state_dict(torch.load(resume_path, map_location=map_location)[
                                                   'model_state'], strict=False)
                    print('load params from ', resume_path)
                elif '.pth' in resume_path:
                    try:
                        self.model.load_state_dict(
                            torch.load(resume_path, map_location=map_location), strict=True)
                    except:
                        print('fail to load, loose the constraint')
                        self.model.load_state_dict(
                            torch.load(resume_path, map_location=map_location), strict=False)
                    print('load params from ', resume_path)

                else:
                    raise NotImplementedError
            else:
                print('can not find checkpoint under {}'.format(resume_path))

    def forward(self, input):
        pred = self.model.forward(input)
        return pred

    def eval(self):
        if self.use_ema:
            # First save original parameters before replacing with EMA version
            self.ema.store(self.model.parameters())
            # Copy EMA parameters to model
            self.ema.copy_to(self.model.parameters())
        self.model.eval()

    def get_loss(self, pred, targets=None, loss_type='cross_entropy'):
        if not targets is None:
            loss = self.basic_loss_fn(pred, targets, loss_type=loss_type)
        else:
            loss = 0.
        self.loss = loss
        return self.loss

    def train(self, if_testing=False):
        if not if_testing:
            self.model.train()
            if self.use_ema:
                self.ema.restore(self.model.parameters())
        else:
            self.eval()

    def reset_optimizers(self):
        self.optimizer.zero_grad()

    def set_optmizers(self, name='adam'):
        if name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=0.0005)
            self.scheduler = None
        elif name == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(
            ), lr=self.lr, momentum=0.99, weight_decay=0.0005)
            self.scheduler = get_scheduler(
                self.optimizer, lr_policy='step', lr_decay_iters=50)
        else:
            raise NotImplementedError

        if self.use_ema:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=0.995)
        else:
            self.ema = None

    def optimize_params(self):
        self.optimizer.step()
        if self.use_ema:
            self.ema.update(self.model.parameters())

    def reset_loss(self):
        self.loss = 0.

    def set_running_metric(self):
        running_metric = runningScore(n_classes=self.num_classes)
        return running_metric

    def predict(self, input):
        gc.collect()  # collect garbage
        self.eval()
        # with torch.no_grad():
        output = self.model.forward(input)
        probs = torch.softmax(output, dim=1)
        torch.cuda.empty_cache()

        return probs

    def MC_predict(self, input, n_times=5, decoder_dropout=0.1, disable_bn=False):
        assert n_times >= 1
        # use MC dropout to get ensembled prediction
        # enable dropout
        if self.decoder_dropout is None or self.decoder_dropout != decoder_dropout:
            self.decoder_dropout = decoder_dropout
            self.model = self.get_network_from_model_library(self.network_type)
            self.init_model(self.network_type)
            if self.use_gpu:
                self.model.cuda()
        self.model.train()
        # fix batch norm
        if not disable_bn:
            for module in self.model.modules():
                # print(module)
                if isinstance(module, nn.BatchNorm2d):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()
        # mc sampling
        probs_list = []
        for i in range(n_times):
            image = input.detach()
            output = self.model.forward(image)
            probs_i = torch.softmax(output, dim=1)

            probs_list.append(probs_i)
            torch.cuda.empty_cache()
        mean_probs = sum(probs_list)/len(probs_list)
        return mean_probs, probs_list

    def evaluate(self, input, targets_npy):
        '''
        evaluate the model performance

        :param input: 4-d tensor input: NCHW
        :param targets_npy: numpy ndarray: N*H*W
        :param running_metric: runnning metric for evaluatation
        :return:
        '''
        gc.collect()  # collect garbage
        self.train(if_testing=True)
        pred = self.predict(input)
        pred_npy = pred.max(1)[1].cpu().numpy()
        del pred
        self.running_metric.update(
            label_trues=targets_npy, label_preds=pred_npy)
        self.cur_eval_images = input.data.cpu().numpy()[:, 0, :, :]
        del input
        self.cur_eval_predicts = pred_npy
        self.cur_eval_gts = targets_npy  # N*H*W

        return pred_npy

    def basic_loss_fn(self, pred, target, loss_type='cross_entropy'):
        'this function contains basic segmentation losses in the supervised setting'
        l = 0.
        # print (cls_weight)
        loss_type = loss_type.strip()
        if loss_type == 'cross_entropy':
            l = cross_entropy_2D(pred, target)
        elif loss_type == 'dice':
            l = dice(pred, target)
        else:
            raise NotImplementedError

        return l

    def save_model(self, save_dir, epoch_iter, model_prefix=None):
        if model_prefix is None:
            model_prefix = self.network_type
        epoch_path = join(save_dir, *[str(epoch_iter), 'checkpoints'])
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)

        torch.save(self.model.state_dict(),
                   join(epoch_path, model_prefix + '$' + 'SAX' + '$' + '_Segmentation' + '.pth'))

    def save_current_results(self, save_name='predict.npy'):
        raise NotImplementedError

    def save_testing_images_results(self, save_dir, epoch_iter, max_slices=10, file_name='Seg_plots.png'):
        if epoch_iter == '':
            epoch_result_path = join(save_dir, 'predict')
        else:
            if isinstance(epoch_iter, int):
                epoch_result_path = join(
                    save_dir, *[str(epoch_iter), 'testing_segmentation_results'])
            if isinstance(epoch_iter, str):
                epoch_result_path = join(
                    save_dir, *[epoch_iter, 'testing_segmentation_results'])

        if not os.path.exists(epoch_result_path):
            os.makedirs(epoch_result_path)
        gts = self.cur_eval_gts
        predicts = self.cur_eval_predicts
        images = self.cur_eval_images

        total_list = []
        init = True
        labels = []

        for subj_index in range(min(max_slices, gts.shape[0])):
            # for each subject
            alist = []
            temp_gt_A = gts[subj_index]
            temp_img_A = images[subj_index]
            temp_pred_A = predicts[subj_index]

            # add image and gt
            alist.append(temp_img_A)
            alist.append(temp_gt_A)
            alist.append(temp_pred_A)

            if init:
                labels.append('Input')
                labels.append('GT')
                labels.append('Predict')

            init = False

            total_list.append(alist)

        save_list_results_as_png(total_list,
                                 save_full_path=join(epoch_result_path,
                                                     file_name),
                                 labels=labels)
