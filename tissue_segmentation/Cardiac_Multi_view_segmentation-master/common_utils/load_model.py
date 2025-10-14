import torch
import os
def resume_model_from_file(file_path):
    start_epoch=1
    optimizer_state=None
    state_dict=None
    checkpoint=None
    assert os.path.isfile(file_path)
    if '.pkl' in file_path:
        print("Loading models and optimizer from checkpoint '{}'".format(file_path))
        checkpoint = torch.load(file_path)
        for k,v in checkpoint.items():
            if k=='model_state':
                state_dict=checkpoint['model_state']
            if k=='optimizer_state':
                optimizer_state=checkpoint['optimizer_state']
            if k=='epoch':
                start_epoch = int(checkpoint['epoch'])
        print("Loaded checkpoint '{}' (epoch {})"
              .format(file_path, checkpoint['epoch']))
    elif '.pth' in file_path:
        print("Loading models and optimizer from checkpoint '{}'".format(file_path))
        state_dict = torch.load(file_path)
        start_epoch=int(file_path.split('.')[0].split('_')[-1]) ##restore training procedure.
    else:
        raise NotImplementedError

    return {'start_epoch':start_epoch,
            'optimizer_state':optimizer_state,
            'state_dict':state_dict,
            'checkpoint':checkpoint
            }

def restoreOmega(path,model,optimizer=None):
    checkpoint = resume_model_from_file(file_path=path)
    state_dict = checkpoint['state_dict']
    start_epoch = checkpoint['start_epoch']
    model.load_state_dict(state_dict, strict=False)
    optimizer_state = checkpoint['optimizer_state']
    if not (optimizer_state is None) and (not optimizer is None):
        try:
            optimizer.load_state_dict(optimizer_state)
        except:
            pass
    return model,optimizer,start_epoch
def save_model_to_file(model_name,model, epoch, optimizer,save_path):
    state_dict= model.module.state_dict()  if isinstance(model,torch.nn.DataParallel) else model.state_dict()
    state = {'model_name': model_name,
             'epoch': epoch + 1,
             'model_state': state_dict,
             'optimizer_state': optimizer.state_dict()
             }
    torch.save(state, save_path)

def gen_overlay(img,attention):
    import numpy as np
    import cv2
    '''
    2D
    :param image: 2D
    :param attention: 2D
    :return: 2D
    '''
    img = img[:, :, np.newaxis]
    # img=(img*255.0)
    img = np.repeat(img, axis=2, repeats=3)
    height, width, _ = img.shape
    img=(img-img.min())/(img.max()-img.min())*255
    img=img.astype(np.uint8)
    #attention= np.expand_dims(attention, axis=2)
   # attention= np.repeat(attention, axis=2, repeats=3)


    attention=(attention-attention.min())/(attention.max()-attention.min())*255
    attention=attention.astype(np.uint8)
    heatmap = cv2.applyColorMap(attention,cv2.COLORMAP_JET)
    cam=cv2.addWeighted(img, 1, heatmap, 1, 0)

    # cam = heatmap + np.float32(img)
    #cam = cam / np.max(cam)
    return cam#np.uint8(cam*255)
def save_npy2image(data,file_dir,name):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    filepath=os.path.join(file_dir,name+'.png')
    import scipy.misc
    scipy.misc.imsave(filepath, data)
