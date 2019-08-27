# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.
# based on github.com/ClementPinard/SfMLearner-Pytorch

import torch
from torch import nn
from torch.autograd import Variable
from inverse_warp import inverse_warp, flow_warp
from ssim import ssim
from process_functions import depth_occlusion_masks,occlusion_masks

from utils import robust_l1,logical_or,weighted_binary_cross_entropy
from utils import tensor2array

import matplotlib.pyplot as plt

#loss1 E_R recunstruction loss
def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose, rotation_mode='euler', padding_mode='zeros', lambda_oob=0, qch=0.5, wssim=0.5):
    def one_scale(depth, explainability_mask, occ_masks):
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)

        weight = 1.
        #
        for i, ref_img in enumerate(ref_imgs_scaled):#ref_imgs_scaled: list with 4 items( ref dimention)
            current_pose = pose[:, i]

            ref_img_warped = inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)

            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)#[4,1,h,w]

            diff = (tgt_img_scaled - ref_img_warped) * valid_pixels#[4,3,h,w]

            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels#ssim(a,b)返回同样shape的c,按元素，越相似越接近1， 否则最低为0

            oob_normalization_const = valid_pixels.nelement()/valid_pixels.sum()#avg, 根据monodepth改成min??

            assert((oob_normalization_const == oob_normalization_const).item() == 1)


            if explainability_mask is not None:
                diff = diff * (1 - occ_masks[:,i:i+1])* explainability_mask[:,i:i+1].expand_as(diff)
                ssim_loss = ssim_loss * (1 - occ_masks[:,i:i+1])* explainability_mask[:,i:i+1].expand_as(ssim_loss)
            else:
                diff = diff *(1-occ_masks[:,i:i+1]).expand_as(diff)
                ssim_loss = ssim_loss*(1-occ_masks[:,i:i+1]).expand_as(ssim_loss)

            reconstruction_loss +=  (1- wssim)*weight*oob_normalization_const*(robust_l1(diff, q=qch) + wssim*ssim_loss.mean()) + lambda_oob*robust_l1(1 - valid_pixels, q=qch)
            assert((reconstruction_loss == reconstruction_loss).item() == 1)
            #weight /= 2.83
        return reconstruction_loss


    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    loss = 0
    for d, mask in zip(depth, explainability_mask):
        occ_masks = depth_occlusion_masks(d, pose, intrinsics, intrinsics_inv)
        loss += one_scale(d, mask, occ_masks)
    return loss


def photometric_reconstruction_loss_robust(tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose, rotation_mode='euler', padding_mode='zeros', lambda_oob=0, qch=0.5, wssim=0.5):
    def one_scale(depth, explainability_mask, occ_masks):
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)

        weight = 1.

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped = inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * valid_pixels
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels#ssim(a,b)返回同样shape的c,按元素，越相似越接近1， 否则最低为0
            oob_normalization_const = valid_pixels.nelement()/valid_pixels.sum()

            assert((oob_normalization_const == oob_normalization_const).item() == 1)

            if explainability_mask is not None:
                diff = diff * (1 - occ_masks[:,i:i+1])* explainability_mask[:,i:i+1].expand_as(diff)
                ssim_loss = ssim_loss * (1 - occ_masks[:,i:i+1])* explainability_mask[:,i:i+1].expand_as(ssim_loss)
            else:
                diff = diff *(1-occ_masks[:,i:i+1]).expand_as(diff)
                ssim_loss = ssim_loss*(1-occ_masks[:,i:i+1]).expand_as(ssim_loss)

            reconstruction_loss +=  (1- wssim)*weight*oob_normalization_const*(robust_l1(diff, q=qch) + wssim*ssim_loss.mean()) + lambda_oob*robust_l1(1 - valid_pixels, q=qch)
            assert((reconstruction_loss == reconstruction_loss).item() == 1)
            #weight /= 2.83
        return reconstruction_loss

    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    loss = 0
    for d, mask in zip(depth, explainability_mask):
        occ_masks = depth_occlusion_masks(d, pose, intrinsics, intrinsics_inv)
        loss += one_scale(d, mask, occ_masks)
    return loss


#loss2 E_M
def explainability_loss(mask):
    '''
    mask 面积越大, 损失越大
    :param mask:
    :return:
    '''
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = torch.ones(1).expand_as(mask_scaled).type_as(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss

#loss_3 E_S smooth loss
def smooth_loss(pred_disp):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_disp) not in [tuple, list]:
        pred_disp = [pred_disp]

    loss = 0
    weight = 1.

    for scaled_disp in pred_disp:
        dx, dy = gradient(scaled_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # 2sqrt(2)
    return loss


#loss4 E_f, flow_loss
def photometric_flow_loss(tgt_img, ref_imgs, flows, explainability_mask, lambda_oob=0, qch=0.5, wssim=0.5):
    '''
        call: occlusion mask:通过光流反解ref，计算差异性损失

    aug:
    flows:[flow_fwd,flow_bwd]
        flow_fwd/list
            list
                |--0:tensor:[4,2,128,512]
                ...
                |--5:tensor:[4,2,4,16]
            ....
    explainability_mask:flow_exp_mask
        |--0:tensor:[4,2,128,512]
        ....


    '''
    def one_scale(explainability_mask, occ_masks, flows):
        assert(explainability_mask is None or flows[0].size()[2:] == explainability_mask.size()[2:])
        assert(len(flows) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = flows[0].size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]

        weight = 1.

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_flow = flows[i]

            ref_img_warped = flow_warp(ref_img, current_flow)#fomulate 48 w_c
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * valid_pixels
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels
            oob_normalization_const = valid_pixels.nelement()/valid_pixels.sum()

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)
                ssim_loss = ssim_loss * explainability_mask[:,i:i+1].expand_as(ssim_loss)

            if occ_masks is not None:
                diff = diff *(1-occ_masks[:,i:i+1]).expand_as(diff)
                ssim_loss = ssim_loss*(1-occ_masks[:,i:i+1]).expand_as(ssim_loss)

            reconstruction_loss += (1- wssim)*weight*oob_normalization_const*(robust_l1(diff, q=qch) + wssim*ssim_loss.mean()) + lambda_oob*robust_l1(1 - valid_pixels, q=qch)
            #weight /= 2.83
            assert((reconstruction_loss == reconstruction_loss).item() == 1)

        return reconstruction_loss

    if type(flows[0]) not in [tuple, list]:#flows[0] is flow_fwd , a list or no
        if explainability_mask is not None:
            explainability_mask = [explainability_mask]
        flows = [[uv] for uv in flows]

    loss = 0
    for i in range(len(flows[0])):#根据尺度遍历scales
        flow_at_scale = [uv[i] for uv in flows]#flow_at_sacle:list(2):item:tensor:[4,2,128/i^2,512/i^2],2 是向量图的缘故
        occ_mask_at_scale_bw, occ_mask_at_scale_fw  = occlusion_masks(flow_at_scale[0], flow_at_scale[1])#0:fwd;1:bwd
        #occ_mask-at_scale_bw.shape[b,h,w]
        occ_mask_at_scale = torch.stack((occ_mask_at_scale_bw, occ_mask_at_scale_fw), dim=1)
        # occ_mask_at_scale = None
        loss += one_scale(explainability_mask[i], occ_mask_at_scale, flow_at_scale)

    return loss


#loss_5 E_C 修改过one_scale loss func
def consensus_depth_flow_mask(explainability_mask, census_mask_bwd, census_mask_fwd, exp_masks_bwd_target, exp_masks_fwd_target, THRESH, wbce):
    # Loop over each scale

    def one_scale(explainability_mask, census_mask_bwd, census_mask_fwd, exp_masks_bwd_target, exp_masks_fwd_target, THRESH, wbce):
    #for i in range(len(explainability_mask)):
        #exp_mask_one_scale = explainability_mask
        census_mask_fwd_one_scale = (census_mask_fwd < THRESH).type_as(explainability_mask).prod(dim=1, keepdim=True)
        census_mask_bwd_one_scale = (census_mask_bwd < THRESH).type_as(explainability_mask).prod(dim=1, keepdim=True)
        #census_mask_bwd_one_scale:tensor[b,1,h,w]
        #Using the pixelwise consensus term
        exp_fwd_target_one_scale = exp_masks_fwd_target
        exp_bwd_target_one_scale = exp_masks_bwd_target
        census_mask_fwd_one_scale = logical_or(census_mask_fwd_one_scale, exp_fwd_target_one_scale)
        census_mask_bwd_one_scale = logical_or(census_mask_bwd_one_scale, exp_bwd_target_one_scale)

        # OR gate for constraining only rigid pixels
        # exp_mask_fwd_one_scale = (exp_mask_one_scale[:,2].unsqueeze(1) > 0.5).type_as(exp_mask_one_scale)
        # exp_mask_bwd_one_scale = (exp_mask_one_scale[:,1].unsqueeze(1) > 0.5).type_as(exp_mask_one_scale)
        # census_mask_fwd_one_scale = 1- (1-census_mask_fwd_one_scale)*(1-exp_mask_fwd_one_scale)
        # census_mask_bwd_one_scale = 1- (1-census_mask_bwd_one_scale)*(1-exp_mask_bwd_one_scale)

        census_mask_fwd_one_scale = Variable(census_mask_fwd_one_scale.data, requires_grad=False)
        census_mask_bwd_one_scale = Variable(census_mask_bwd_one_scale.data, requires_grad=False)

        rigidity_mask_combined = torch.cat((census_mask_bwd_one_scale, census_mask_bwd_one_scale,
                        census_mask_fwd_one_scale, census_mask_fwd_one_scale), dim=1)
        return weighted_binary_cross_entropy(explainability_mask, rigidity_mask_combined.type_as(explainability_mask), [wbce, 1-wbce])

    assert (len(explainability_mask) == len(census_mask_bwd))
    assert (len(explainability_mask) == len(census_mask_fwd))
    loss = 0.

    if type(explainability_mask) not in [tuple, list]:
        return one_scale(explainability_mask, census_mask_bwd, census_mask_fwd, exp_masks_bwd_target, exp_masks_fwd_target, THRESH, wbce)
    else:
        for i in range(len(explainability_mask)):
            loss+=one_scale(explainability_mask[i], census_mask_bwd[i], census_mask_fwd[i], exp_masks_bwd_target[i], exp_masks_fwd_target[i], THRESH, wbce)
        return loss







#note use
def gaussian_explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        loss += torch.exp(-torch.mean((mask_scaled-0.5).pow(2))/0.15)
    return loss






def edge_aware_smoothness_loss(img, pred_disp):
    def gradient_x(img):
      gx = img[:,:,:-1,:] - img[:,:,1:,:]
      return gx

    def gradient_y(img):
      gy = img[:,:,:,:-1] - img[:,:,:,1:]
      return gy

    def get_edge_smoothness(img, pred):
      pred_gradients_x = gradient_x(pred)
      pred_gradients_y = gradient_y(pred)

      image_gradients_x = gradient_x(img)
      image_gradients_y = gradient_y(img)

      weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
      weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

      smoothness_x = torch.abs(pred_gradients_x) * weights_x
      smoothness_y = torch.abs(pred_gradients_y) * weights_y
      return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.

    for scaled_disp in pred_disp:
        b, _, h, w = scaled_disp.size()
        scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
        loss += get_edge_smoothness(scaled_img, scaled_disp)
        weight /= 2.3   # 2sqrt(2)

    return loss





