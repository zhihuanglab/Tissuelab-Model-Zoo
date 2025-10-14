import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize as ColorNormalize
from mpl_toolkits.axes_grid1 import make_axes_locatable # plotting
from PIL import Image
from skimage import transform
import cv2
from scipy.ndimage.interpolation import map_coordinates

import torchvision.utils as vutils

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def array2image(array,if_mask=False):
    '''
    input H*W
    :param array:
    :return: nd aaray 3*H*W

    '''
    old_array=array.copy()
    if if_mask:

        output = np.asarray(array, dtype=np.uint8)
        color_image = colorize_mask(output)
        aligned_image = Image.new("RGB", (old_array.shape[0], old_array.shape[1]))
        aligned_image.paste(color_image, (0, 0))
        image_numpy =  np.asarray(aligned_image)

        image_numpy = np.transpose(image_numpy, (2, 0, 1))
        return image_numpy

    normed_array = (((array - array.min()) / (array.max() - array.min())) * 255).astype(np.uint8)

    I = Image.fromarray(normed_array[ :, :])
    aligned_image = Image.new("RGB", (normed_array.shape[0], normed_array.shape[1]))
    aligned_image.paste(I, (0, 0))
    image_numpy = np.array(aligned_image)
    image_numpy=np.transpose(image_numpy,(2,0,1))
    return image_numpy


def show_image(writer,image_npy,i_iter,name='image'):
    '''

    :param writer: tensorboardX writer object
    :param image_npy:npy array N*C*H*W
    :param i_iter:
    :param name:
    :return:
    '''
    image_new = np.zeros((image_npy.shape[0], 3, image_npy.shape[2], image_npy.shape[3]))
    for i in range(image_npy.shape[0]):
        slice = array2image(image_npy[i,0,])
        image_new[i] = slice
    image_new_tensor = torch.from_numpy(image_new)

    x=vutils.make_grid(image_new_tensor,nrow=4,normalize=True,scale_each=True)
    writer.add_image(name,x,i_iter)

def show_label_image(writer, pred_npy, i_iter, name='label'):
    pred=pred_npy
    pred_new = np.zeros((pred.shape[0], 3, pred.shape[1], pred.shape[2]))
    for i in range(pred_new.shape[0]):
        slice = array2image(pred[i], if_mask=True)
        pred_new[i] = slice
    pred_tensor = torch.from_numpy(pred_new)
    x = vutils.make_grid(pred_tensor, nrow=4)
    writer.add_image(name, x, i_iter)

def flow_legend():
    """
    show quiver plot to indicate how arrows are colored in the flow() method.
    https://stackoverflow.com/questions/40026718/different-colours-for-arrows-in-quiver-plot
    """
    ph = np.linspace(0, 2 * np.pi, 13)
    x = np.cos(ph)
    y = np.sin(ph)
    u = np.cos(ph)
    v = np.sin(ph)
    colors = np.arctan2(u, v)

    norm = ColorNormalize()
    norm.autoscale(colors)
    # we need to normalize our colors array to match it colormap domain
    # which is [0, 1]

    colormap = cm.winter

    plt.figure(figsize=(6, 6))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.quiver(x, y, u, v, color=colormap(norm(colors)), angles='xy', scale_units='xy', scale=1)
    plt.show()


def vis_flow(slices_in,  # the 2D slices
         titles=None,  # list of titles
         cmaps=None,  # list of colormaps
         width=15,  # width in in
         img_indexing=True,  # whether to match the image view, i.e. flip y axis
         grid=False,  # option to plot the images in a grid or a single row
         show=True,  # option to actually show the plot (plt.show())
         scale=1):  # note quiver essentially draws quiver length = 1/scale
    '''
    ref: https://github.com/voxelmorph/voxelmorph/blob/8b5e96af9cdca7474b4ee658d57f70cd4dadba22/ext/neuron/neuron/plot.py
    plot a grid of flows (2d+2 images)
    '''

    # input processing
    nb_plots = len(slices_in)
    for slice_in in slices_in:
        assert len(slice_in.shape) == 3, 'each slice has to be 3d: 2d+2 channels'
        assert slice_in.shape[-1] == 2, 'each slice has to be 3d: 2d+2 channels'

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    if img_indexing:
        for si, slc in enumerate(slices_in):
            slices_in[si] = np.flipud(slc)

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    scale = input_check(scale, nb_plots, 'scale')

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        u, v = slices_in[i][..., 0], slices_in[i][..., 1]
        colors = np.arctan2(u, v)
        colors[np.isnan(colors)] = 0
        norm = ColorNormalize()
        norm.autoscale(colors)
        if cmaps[i] is None:
            colormap = cm.winter
        else:
            raise Exception("custom cmaps not currently implemented for plt.flow()")

        # show figure
        ax.quiver(u, v,
                  color=colormap(norm(colors).flatten()),
                  angles='xy',
                  units='xy',
                  scale=scale[i])
        ax.axis('equal')

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)
    plt.tight_layout()

    if show:
        plt.show()

    return (fig, axs, plt)

# Define function to draw a grid
def plt_deformed_grid(deformation_field_list, grid_size=10,**kwargs):
    '''

    :param deformation_field: a list of deformation field: [N*H*W*2] [dx,dy]
    :param grid_size:
    :param kwargs:
    :return:
    '''
    fig, axs = plt.subplots(1, len(deformation_field_list),squeeze=False)
    for ind, deformation_field in enumerate(deformation_field_list):
        gridy = deformation_field[:,:,1]
        gridx = deformation_field[:,:,0]
        gridy=np.array(gridy[::grid_size,::grid_size])
        gridx=np.array(gridx[::grid_size,::grid_size])
        # ## plot horizontal lines
        # gridx= np.transpose(gridx)
        # gridy= np.transpose(gridy)
        def get_colors (x,y):
            # output_color='r'
            colors =x*2+y*2
            colors[np.isnan(colors)] = 0
            norm = ColorNormalize()
            norm.autoscale(colors)
            colormap = cm.hsv

            output_color= colormap(norm(colors).flatten())
            # print (output_color.shape)
            return output_color

        for i in range(gridx.shape[0]):
            x = gridx[i, :]
            y = gridy[i, :][::-1]
            axs[0,ind].plot(x,y,linewidth=4,color=get_colors(x,y)[0])

        ## plot vertical lines
        for i in range(gridy.shape[1]):
            x = gridx[:, i]
            y = gridy[:, i][::-1]
            axs[0,ind].plot(x,y,linewidth=4,color=get_colors(x,y)[0])

    width=100
    rows=1
    cols= len(deformation_field_list)
    fig.set_size_inches(width, rows / cols * width)
    plt.tight_layout()

    return fig,axs,plt




if __name__ == '__main__':
    ## plot deformation field using quiver.
    import seaborn as sns
    sns.set()
    from scipy.ndimage.filters import gaussian_filter

    import numpy as np
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    image_height=80
    image_width = 80
    bs =1
    angle = 100/(180.0)*np.pi
    rot_theta = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0]
    ], dtype=torch.float)
    rot_theta = rot_theta.view(-1, 2, 3)
    rot_theta = rot_theta.repeat(bs, 1, 1)

    ## identity
    i_theta = torch.tensor([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=torch.float)
    i_theta = i_theta.view(-1, 2, 3)
    i_theta = i_theta.repeat(bs, 1, 1)

    rot_grid = torch.affine_grid_generator(rot_theta, size=(bs, 1, image_height,image_width),
                                       align_corners=False)  ##output shape (N, H_\text{out}, W_\text{out}, 2)

    id_grid = torch.affine_grid_generator(i_theta, size=(bs, 1, image_height, image_width),
                                       align_corners=False)  ##output shape (N, H_\text{out}, W_\text{out}, 2) [dy,dx]
    random_state = np.random.RandomState(None)
    rot_grid_npy = rot_grid.numpy()[0]## H*W*2
    id_grid_npy = id_grid.numpy()[0] ## H*W*2
    empty_deform_npy = np.zeros(id_grid_npy.shape) ## H*W*2
    dx = gaussian_filter((random_state.rand(image_height,image_width,1) * 2 - 1), 10, mode="constant", cval=0) * 18
    dy = gaussian_filter((random_state.rand(image_height,image_width,1) * 2 - 1), 10, mode="constant", cval=0) *18
    dz = np.zeros_like(dx)
    deformed_grid = plt_deformed_grid([np.concatenate((dx,dy),2)+id_grid_npy],grid_size=5)
    # plt.imshow(deformed_grid)

    local_deform_grid = np.zeros((image_height,image_width,2))
    local_deform_grid[4:16,4:16] = rot_grid_npy[4:16,4:16]
    slices_in= [empty_deform_npy,id_grid_npy,rot_grid_npy[:,:,::-1],(rot_grid_npy-id_grid_npy)[:,:,::-1],(rot_grid_npy**2)[:,:,::-1]]
    vis_flow(slices_in=slices_in,
         img_indexing=True,  # whether to match the image view, i.e. flip y axis
         grid=True,  # option to plot the images in a grid or a single row
         show=True,  # option to actually show the plot (plt.show())
         scale=1,
         )
    flow_legend()

    input_image =np.load('/vol/medic01/users/cc215/Dropbox/projects/DeformADA/Data/SHUO_data/ED/4193118_img.npy')


    ### generate random deformation field
    ## control points -based
    num_points=image_width
    sigma =5
    alpha =1
    threshhold=1.

    ## initialization:
    dx = np.random.randn(num_points,num_points)
    dy = np.random.randn(num_points,num_points)

    ## smooth the deformation
    for i in range(1000):
        dx = gaussian_filter(dx, sigma=sigma, mode='constant', cval=0)*alpha
        dy = gaussian_filter(dy, sigma=sigma, mode='constant', cval=0)*alpha
        std_deviation=np.sqrt(np.sum(dx**2+dy**2))
        if np.abs(std_deviation)<threshhold:
            print ('iter:',i)
            print( 'sum of deviation:', np.sum(dx ** 2 + dy ** 2))
            print ('stop')
            break
    ## rescale the deformation to the image space
    random_dx = transform.resize(dx, output_shape=(image_width, image_height), order=3, mode='reflect')
    random_dy = transform.resize(dy, output_shape=(image_width, image_height), order=3, mode='reflect')

    ## gen grid
    x, y = np.meshgrid(np.linspace(-1,1,image_width), np.linspace(-1,1,image_height))  ## image space [0-H]
    grid_x=x+random_dx
    grid_y=y+random_dy

    ## restrict indices to -1,1
    grid_x = np.clip(grid_x,-1,1)
    grid_y = np.clip(grid_y,-1,1)

    indices =np.reshape(grid_x, (image_height,image_width ,1)), np.reshape(grid_y, (image_height,image_width, 1))
    x_new_ind=indices[0]
    y_new_ind=indices[1]

    flow=np.concatenate((x_new_ind,y_new_ind), axis=2)
    ## make it batch-wise
    flow_batch=np.repeat(flow[np.newaxis,:,:,:],repeats=bs,axis=0)
    flow_tensor = torch.from_numpy(flow_batch).float()
    input_tensor = torch.from_numpy(input_image[np.newaxis,np.newaxis,:,:]).float()

    transformed_image = F.grid_sample(input_tensor,flow_tensor,mode='bilinear')
    plt.subplot(141)
    plt.imshow(flow[:,:,0]) ## dx
    plt.colorbar()
    plt.subplot(142)
    plt.imshow(flow[:,:,1])
    plt.colorbar()
    plt.subplot(143)
    plt.imshow(transformed_image.numpy()[0,0])
    plt.subplot(144)
    plt.imshow(input_image-transformed_image.numpy()[0,0])
    plt.show()

    # from libcpab.pytorch import cpab

    # T = cpab(tess_size=[2, 2])


