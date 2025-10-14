import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler
import math


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    code reference: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py

    """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) /
                        (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args: 
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args: 
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone()
                                 for param in parameters
                                 if param.requires_grad]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args: 
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        if len(self.collected_params) > 0:
            for c_param, param in zip(self.collected_params, parameters):
                if param.requires_grad:
                    param.data.copy_(c_param.data)
        else:
            print('did not find any copy, use the original params')


def set_grad(module, requires_grad=False):
    for p in module.parameters():  # reset requires_grad
        p.requires_grad = requires_grad


def makeVariable(tensor, use_gpu=True, type='long', requires_grad=True):
    # conver type
    tensor = tensor.data
    if type == 'long':
        tensor = tensor.long()
    elif type == 'float':
        tensor = tensor.float()
    else:
        raise NotImplementedError

    # make is as Variable
    if use_gpu:
        variable = Variable(tensor.cuda(), requires_grad=requires_grad)
    else:
        variable = Variable(tensor, requires_grad=requires_grad)
    return variable


def get_scheduler(optimizer, lr_policy, lr_decay_iters=5, epoch_count=None, niter=None, niter_decay=None):
    print('lr_policy = [{}]'.format(lr_policy))
    if lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + epoch_count -
                             niter) / float(niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_iters, gamma=0.5)
    elif lr_policy == 'step2':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        print('schedular=plateau')
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, threshold=0.01, patience=5)
    elif lr_policy == 'plateau2':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'step_warmstart':
        def lambda_rule(epoch):
            # print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 100:
                lr_l = 1
            elif 100 <= epoch < 200:
                lr_l = 0.1
            elif 200 <= epoch:
                lr_l = 0.01
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step_warmstart2':
        def lambda_rule(epoch):
            # print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 50:
                lr_l = 1
            elif 50 <= epoch < 100:
                lr_l = 0.1
            elif 100 <= epoch:
                lr_l = 0.01
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:

        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


def spatial_pyramid_pool(previous_conv, batch_size, previous_conv_size, out_bin_sizes):
    """[summary]
    Performs spatial pyramid pooling on a batch of tensors
    Args:
        previous_conv (torch tensor): 4-dim input feature  maps
        batch_size (int):
        previous_conv_size (type):4-dim input feature  maps
        out_bin_sizes (list): a list of window  sizes.

    Returns:
        flattened feature maps with multi-scale information (torch tensor): [bs,f]
    """
    for i in range(0, len(out_bin_sizes)):
        print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_bin_sizes[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_bin_sizes[i]))
        h_pad = (h_wid * out_bin_sizes[i] - previous_conv_size[0] + 1) // 2
        w_pad = (w_wid * out_bin_sizes[i] - previous_conv_size[1] + 1) // 2
        maxpool = nn.MaxPool2d(kernel_size=(h_wid, w_wid), stride=(
            h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if (i == 0):
            spp = x.view(batch_size, -1)
        else:
            spp = torch.cat((spp, x.view(batch_size, -1)), dim=1)
    return spp


'''
https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/4
'''


class GradientReversalFunction(torch.autograd.Function):
    def __init__(self, Lambda):
        super(GradientReversalFunction, self).__init__()
        self.Lambda = Lambda

    def forward(self, input):
        return input.view_as(input)

    def backward(self, grad_output):
        # Multiply gradient by -self.Lambda
        return self.Lambda * grad_output.neg()


class GradientReversalLayer(nn.Module):
    def __init__(self, Lambda, use_cuda=False):
        super(GradientReversalLayer, self).__init__()
        self.Lambda = Lambda
        if use_cuda:
            self.cuda()

    def forward(self, input):
        return GradientReversalFunction(self.Lambda)(input)

    def change_lambda(self, Lambda):
        self.Lambda = Lambda


def calc_gradient_penalty(netD, lamda, real_data, fake_data, gpu=0):
    from torch import autograd
    # print ("real_data: ", real_data.size())
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous(
    ).view(batch_size, real_data.size(1), real_data.size(2),  real_data.size(3))
    if gpu is not None:
        alpha = alpha.cuda(gpu)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if gpu is not None:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if gpu is not None else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamda
    return gradient_penalty


def encode(label_map, n_classes):
    '''
    convert input label into one-hot tensor
    return onehot label
    :param label: batch_size*1*target_h*target_w
    :return:label:batch_size*n_classes*target_h*target_w
    '''
    # create one-hot vector for label map
    label_map = label_map[:, None, :, :]
    size = label_map.size()
    print(size)
    oneHot_size = (size[0], n_classes, size[2], size[3])
    input_label = torch.zeros(torch.Size(oneHot_size)).float().cuda()
    label_map = Variable(label_map)
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    return input_label


def gram_matrix_2D(y):
    '''
    give torch 4d tensor, calculate Gram Matrix
    :param y:
    :return:
    '''
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to a fixed number"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_scheduler(optimizer, lr_policy, lr_decay_iters=5, epoch_count=None, niter=None, niter_decay=None):
    print('lr_policy = [{}]'.format(lr_policy))
    if lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + epoch_count -
                             niter) / float(niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_iters, gamma=0.5)
    elif lr_policy == 'step2':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        print('schedular=plateau')
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, threshold=0.01, patience=5)
    elif lr_policy == 'plateau2':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'step_warmstart':
        def lambda_rule(epoch):
            # print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 100:
                lr_l = 1
            elif 100 <= epoch < 200:
                lr_l = 0.1
            elif 200 <= epoch:
                lr_l = 0.01
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step_warmstart2':
        def lambda_rule(epoch):
            # print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 50:
                lr_l = 1
            elif 50 <= epoch < 100:
                lr_l = 0.1
            elif 100 <= epoch:
                lr_l = 0.01
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:

        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


def cal_cls_acc(pred, gt):
    '''
    input tensor
    :param pred: network output N*n_classes
    :param gt: ground_truth N [labels_id]
    :return: float acc
    '''
    pred_class = pred.data.max(1)[1].cpu()
    sum = gt.cpu().eq(pred_class).sum()
    count = gt.size(0)
    return sum, count


def cal_statistic_loss(featuremaps):
    batch_size = featuremaps[0].size(0)
    style_loss = 0.
    for f in featuremaps:
        level_1 = f  # batch*feature_stats
        loss = 0.
        std_erro = 0.
        mean_erro = 0.
        for i in range(batch_size):
            instance_f = f[i]  # f_n*h*w
            instance_f_view = instance_f.view(1, instance_f.size(
                0), instance_f.size(1)*instance_f.size(2))
            target_std = torch.std(instance_f_view, 2, unbiased=False).view(-1)
            target_mean = torch.mean(
                instance_f_view, 2, keepdim=False).view(-1)
            if i == 0:
                prev_std = target_std
                prev_mean = target_mean
            else:
                std_erro += torch.sum(torch.abs(target_std-prev_std))
                mean_erro += torch.sum(torch.abs(target_mean-prev_mean))
                print(mean_erro)
                print(mean_erro)
        loss += std_erro+mean_erro
    style_loss += loss
    return style_loss / (1.0 * batch_size)


def cal_style_loss(featuremaps, weight):
    '''
    a list of feature maps
    :param featuremaps: [[N_batch*n_kernels*feature_x*feature_y]]
    :return:
    '''
    batch_size = featuremaps[0].size(0)
    print('bn:', batch_size)

    style_loss = 0.
    if batch_size == 4:

        for i, f in enumerate(featuremaps):
            level_1 = f  # batch*feature_stats
            loss = 0.
            # for i in range(batch_size):
            s1 = 0
            s2 = 1
            t1 = 2
            t2 = 3
            # s1, s2 = i, (i + 1) % batch_size
            # t1, t2 = s1 + batch_size, s2 + batch_size
            loss += gram_loss(level_1[[s1]], level_1[[s2]]) + \
                gram_loss(level_1[[t1]], level_1[[t2]])
            loss += (gram_loss(level_1[[s1]], level_1[[t2]]) +
                     gram_loss(level_1[[s2]], level_1[[t1]]))
            loss += (gram_loss(level_1[[s2]], level_1[[t2]]) +
                     gram_loss(level_1[[s1]], level_1[[t1]]))

            style_loss += weight[i]*loss
    else:
        return 0.
    return 2.0*style_loss/(1.0*batch_size)


def cal_fast_style_loss(featuremaps, weight=(0.2, 0.5, 0.3)):
    '''
    a list of feature maps
    :param featuremaps: [[N_batch*n_kernels*feature_x*feature_y]]
    :return:
    '''
    batch_size = featuremaps[0].size(0)//2
    style_loss = 0.
    for f in featuremaps:
        level_1 = f  # batch*feature_stats
        loss = 0.
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += gram_loss(level_1[[s1]], level_1[[s2]]) + \
                gram_loss(level_1[[t1]], level_1[[t2]])
            loss -= (gram_loss(level_1[[s1]], level_1[[t2]]) +
                     gram_loss(level_1[[s2]], level_1[[t1]]))
        style_loss += loss
    return 2.0 * style_loss / (1.0 * batch_size)


def gram_loss(feature_1, feature_2):
    gram1 = gram_matrix_2D(feature_1)
    gram2 = gram_matrix_2D(feature_2)
    return F.mse_loss(gram1, gram2)
