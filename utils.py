from __future__ import division
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from path import Path
from collections import OrderedDict
import datetime

epsilon = 1e-8

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

def save_checkpoint(save_path, dispnet_state, posenet_state, masknet_state, flownet_state,is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'posenet', 'masknet', 'flownet']
    states = [dispnet_state, posenet_state, masknet_state, flownet_state]
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


#process_function and process_function using

def logical_or(a, b):
    return 1 - (1 - a)*(1 - b)

def robust_l1_per_pix(x, q=0.5, eps=1e-2):
    x = torch.pow((x.pow(2) + eps), q)
    return x
def robust_l1(x, q=0.5, eps=1e-2):
    x = torch.pow((x.pow(2) + eps), q)
    x = x.mean()
    return x

def spatial_normalize(disp):
    _mean = disp.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    disp = disp / _mean
    return disp


def compute_all_epes(gt, rigid_pred, non_rigid_pred, rigidity_mask, THRESH=0.5):
    _, _, h_pred, w_pred = rigid_pred.size()
    _, _, h_gt, w_gt = gt.size()
    rigidity_pred_mask = nn.functional.interpolate(rigidity_mask, size=(h_pred, w_pred), mode='bilinear')
    rigidity_gt_mask = nn.functional.interpolate(rigidity_mask, size=(h_gt, w_gt), mode='bilinear')

    non_rigid_pred = (rigidity_pred_mask<=THRESH).type_as(non_rigid_pred).expand_as(non_rigid_pred) * non_rigid_pred
    rigid_pred = (rigidity_pred_mask>THRESH).type_as(rigid_pred).expand_as(rigid_pred) * rigid_pred
    total_pred = non_rigid_pred + rigid_pred

    gt_non_rigid = (rigidity_gt_mask<=THRESH).type_as(gt).expand_as(gt) * gt
    gt_rigid = (rigidity_gt_mask>THRESH).type_as(gt).expand_as(gt) * gt

    all_epe = compute_epe(gt, total_pred)
    rigid_epe = compute_epe(gt_rigid, rigid_pred)
    non_rigid_epe = compute_epe(gt_non_rigid, non_rigid_pred)
    outliers = outlier_err(gt, total_pred)

    return [all_epe, rigid_epe, non_rigid_epe, outliers]

def flow_diff(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = nn.functional.interpolate(pred, size=(h_gt, w_gt), mode='bilinear')
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    diff = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))

    return diff


def weighted_binary_cross_entropy(output, target, weights=None):
    '''
    #by consensus_depth_flow_mask called
    :param output:
    :param target:
    :param weights:
    :return:
    '''
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output + epsilon)) + \
               weights[0] * ((1 - target) * torch.log(1 - output + epsilon))
    else:
        loss = target * torch.log(output + epsilon) + (1 - target) * torch.log(1 - output + epsilon)

    return torch.neg(torch.mean(loss))

def outlier_err(gt, pred, tau=[3,0.05]):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt, valid_gt = gt[:,0,:,:], gt[:,1,:,:], gt[:,2,:,:]
    pred = nn.functional.interpolate(pred, size=(h_gt, w_gt), mode='bilinear')
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))
    epe = epe * valid_gt

    F_mag = torch.sqrt(torch.pow(u_gt, 2)+ torch.pow(v_gt, 2))
    E_0 = (epe > tau[0]).type_as(epe)
    E_1 = ((epe / (F_mag+epsilon)) > tau[1]).type_as(epe)
    n_err = E_0 * E_1 * valid_gt
    #n_err   = length(find(F_val & E>tau(1) & E./F_mag>tau(2)));
    #n_total = length(find(F_val));
    f_err = n_err.sum()/(valid_gt.sum() + epsilon);
    if type(f_err) == Variable: f_err = f_err.data
    return f_err.item()


def compute_epe(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()

    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = nn.functional.interpolate(pred, size=(h_gt, w_gt), mode='bilinear')
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))

    if nc == 3:
        valid = gt[:,2,:,:]
        epe = epe * valid
        avg_epe = epe.sum()/(valid.sum() + epsilon)
    else:
        avg_epe = epe.sum()/(bs*h_gt*w_gt)

    if type(avg_epe) == Variable: avg_epe = avg_epe.data

    return avg_epe.item()


def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]

