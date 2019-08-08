from __future__ import division
import shutil
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from path import Path
from collections import OrderedDict
import datetime


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0,1,low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0,max_value,resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:,i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array[:,:,:3]
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        if (tensor.size(0) == 3):
            array = 0.5 + tensor.numpy()*0.5
        elif (tensor.size(0) == 2):
            array = tensor.numpy()

    return array

def save_checkpoint(save_path, dispnet_state, posenet_state, masknet_state, flownet_state, optimizer_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'posenet', 'masknet', 'flownet', 'optimizer']
    states = [dispnet_state, posenet_state, masknet_state, flownet_state, optimizer_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix,filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix,filename), save_path/'{}_model_best.pth.tar'.format(prefix))


def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['sequence_length'] = 'seq'
    keys_with_prefix['rotation_mode'] = 'rot_'
    keys_with_prefix['padding_mode'] = 'padding_'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'
    #keys_with_prefix['photo_loss_weight'] = 'p'
    #keys_with_prefix['mask_loss_weight'] = 'm'
    #keys_with_prefix['smooth_loss_weight'] = 's'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp



def flow2rgb(flow_map, max_value):# [2,370,1224]
    '''
        [2,h,w]2[3,h,w]

    :param flow_map:
    :param max_value:
    :return:
    '''
    #eturned Tensor shares the same storage with the original one.
    # In-place modifications on either of them will be seen, and may trigger errors in correctness checks.
    def one_scale(flow_map, max_value):
        flow_map_np = flow_map.detach().cpu().numpy()#??what trick
        _, h, w = flow_map_np.shape#[2,h,w]

        flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')#??? 两幅图中某个位置像素都等于0的置位nan

        rgb_map = np.ones((3,h,w)).astype(np.float32)#占位符
        #normalization
        if max_value is not None:
            normalized_flow_map = flow_map_np / max_value
        else:
            #normalized_flow_map = (flow_map_np-flow_map_np.mean())/np.ndarray.std(flow_map_np)
            normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

        #vector2color coding
        rgb_map[0] += normalized_flow_map[0]
        rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
        rgb_map[2] += normalized_flow_map[1]
        return rgb_map.clip(0,1)#上溢,下溢处理,smaller than 0 become 0, and values larger than 1 become 1, 区间内的值不动


    if type(flow_map) not in [tuple, list]:
        return  one_scale(flow_map,max_value)
    else:
        return [one_scale(flow_map_,max_value) for flow_map_ in flow_map]