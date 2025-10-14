import torch.nn as nn

from model.unet_parts import *
from model.model_utils import  *





class UNet(nn.Module):
    def __init__(self, input_channel, num_classes,feature_scale=1,dropout=None,norm=nn.BatchNorm2d):
        '''

        :param input_channel: input image channel
        :param num_classes: segmentation classes
        :param feature_scale:x: float , number of filters with be adjusted to the 1/x of original Unet number of filters. By default, x=1
        :param dropout: None or a prob between 0 and 1
        :param norm: set norm func batch norm or instance norma
        '''
        if dropout is False or dropout is None:
            dropout=None
        else:
            assert isinstance(dropout,float),'specify dropout rate'
        super(UNet, self).__init__()
        self.inc = inconv(input_channel,64//feature_scale, norm=norm)
        self.down1 = down(64//feature_scale, 128//feature_scale, norm=norm,if_SN=False)
        self.down2 = down(128//feature_scale, 256//feature_scale, norm=norm,if_SN=False)
        self.down3 = down(256//feature_scale, 512//feature_scale, norm=norm,if_SN=False)
        self.down4 = down(512//feature_scale, 512//feature_scale, norm=norm,if_SN=False)
        self.up1 = up(512//feature_scale, 512//feature_scale, 256//feature_scale, norm=norm, dropout=dropout,if_SN=False)
        self.up2 = up(256//feature_scale, 256//feature_scale, 128//feature_scale, norm=norm, dropout=dropout,if_SN=False)
        self.up3 = up(128//feature_scale, 128//feature_scale, 64//feature_scale, norm=norm, dropout=dropout,if_SN=False)
        self.up4 = up(64//feature_scale, 64//feature_scale, 64//feature_scale, norm=norm, dropout=dropout,if_SN=False)

        self.outc = outconv(64//feature_scale, num_classes)
        self.n_classes=num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


    def get_net_name(self):
        return 'unet'

    def fix_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                for k in module.parameters():  ##fix all conv layers
                        k.requires_grad = False
            elif 'outc' in name:
                if isinstance(module,nn.Conv2d):
                    for k in module.parameters():  ##except last layers
                        k.requires_grad = True
            else:
               for k in module.parameters(): ##fix all conv layers
                   k.requires_grad=False


    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer.
        """
        b = []

        b.append(self.inc)
        b.append(self.down1)
        b.append(self.down2)
        b.append(self.down3)
        b.append(self.down4)
        b.append(self.up1)
        b.append(self.up2)
        b.append(self.up3)
        b.append(self.up4)
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification task
        """
        b = []
        b.append(self.outc.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]



class INUnet(UNet):
    """[append an instance normalization layer before Unet]

    Args:
        UNet ([type]): [description]

    Returns:
        [pytorch nn model]: [description]

    Yields:
        [type]: [description]
    """
    def __init__(self, input_channel, num_classes,feature_scale=1,dropout=None,norm=nn.BatchNorm2d):
        super(INUnet,self).__init__(input_channel=input_channel, num_classes=num_classes,feature_scale=feature_scale,dropout=dropout,norm=norm)
        ## with learnable z_score normalization
        self.norm_input = nn.InstanceNorm2d(input_channel,affine=True)   
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
    
    def forward(self,x):
        ## when set to eval, the norm stats stop estimate the mean and std for scaling
        x = self.norm_input(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


if __name__=='__main__':
    model = INUnet(input_channel=1, feature_scale=4,num_classes=4,dropout=None)
    model.eval()
    image = torch.autograd.Variable(torch.randn(2, 1, 224, 224))
    result=model(image)
    print (result.size())
