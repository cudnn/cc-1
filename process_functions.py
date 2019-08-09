import torch
from torch.autograd import Variable
import torch.nn as nn
from inverse_warp import flow_warp, pose2flow

from ssim import  ssim
from utils import epsilon,logical_or,robust_l1_per_pix


def consensus_exp_masks(cam_flows_fwd, cam_flows_bwd, flows_fwd, flows_bwd, tgt_img, ref_img_fwd, ref_img_bwd, wssim, wrig, ws=0.1):
    '''
    #by loss_5 called

    :param cam_flows_fwd:
    :param cam_flows_bwd:
    :param flows_fwd:
    :param flows_bwd:
    :param tgt_img:
    :param ref_img_fwd:
    :param ref_img_bwd:
    :param wssim:
    :param wrig:
    :param ws:
    :return:
    '''
    def one_scale(cam_flow_fwd, cam_flow_bwd, flow_fwd, flow_bwd, tgt_img, ref_img_fwd, ref_img_bwd, ws):
        b, _, h, w = cam_flow_fwd.size()
        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_img_scaled_fwd = nn.functional.adaptive_avg_pool2d(ref_img_fwd, (h, w))
        ref_img_scaled_bwd = nn.functional.adaptive_avg_pool2d(ref_img_bwd, (h, w))

        cam_warped_im_fwd = flow_warp(ref_img_scaled_fwd, cam_flow_fwd)
        cam_warped_im_bwd = flow_warp(ref_img_scaled_bwd, cam_flow_bwd)

        flow_warped_im_fwd = flow_warp(ref_img_scaled_fwd, flow_fwd)
        flow_warped_im_bwd = flow_warp(ref_img_scaled_bwd, flow_bwd)

        valid_pixels_cam_fwd = 1 - (cam_warped_im_fwd == 0).prod(1, keepdim=True).type_as(cam_warped_im_fwd)
        valid_pixels_cam_bwd = 1 - (cam_warped_im_bwd == 0).prod(1, keepdim=True).type_as(cam_warped_im_bwd)
        valid_pixels_cam = logical_or(valid_pixels_cam_fwd, valid_pixels_cam_bwd)  # if one of them is valid, then valid

        valid_pixels_flow_fwd = 1 - (flow_warped_im_fwd == 0).prod(1, keepdim=True).type_as(flow_warped_im_fwd)
        valid_pixels_flow_bwd = 1 - (flow_warped_im_bwd == 0).prod(1, keepdim=True).type_as(flow_warped_im_bwd)
        valid_pixels_flow = logical_or(valid_pixels_flow_fwd, valid_pixels_flow_bwd)  # if one of them is valid, then valid

        cam_err_fwd = ((1-wssim)*robust_l1_per_pix(tgt_img_scaled - cam_warped_im_fwd).mean(1,keepdim=True) \
                    + wssim*(1 - ssim(tgt_img_scaled, cam_warped_im_fwd)).mean(1, keepdim=True))
        cam_err_bwd = ((1-wssim)*robust_l1_per_pix(tgt_img_scaled - cam_warped_im_bwd).mean(1,keepdim=True) \
                    + wssim*(1 - ssim(tgt_img_scaled, cam_warped_im_bwd)).mean(1, keepdim=True))
        cam_err = torch.min(cam_err_fwd, cam_err_bwd) * valid_pixels_cam

        flow_err = (1-wssim)*robust_l1_per_pix(tgt_img_scaled - flow_warped_im_fwd).mean(1, keepdim=True) \
                    + wssim*(1 - ssim(tgt_img_scaled, flow_warped_im_fwd)).mean(1, keepdim=True)
        # flow_err_bwd = (1-wssim)*robust_l1_per_pix(tgt_img_scaled - flow_warped_im_bwd).mean(1, keepdim=True) \
        #             + wssim*(1 - ssim(tgt_img_scaled, flow_warped_im_bwd)).mean(1, keepdim=True)
        # flow_err = torch.min(flow_err_fwd, flow_err_bwd)

        exp_target = (wrig*cam_err <= (flow_err+epsilon)).type_as(cam_err)

        return exp_target

    assert(len(cam_flows_bwd)==len(cam_flows_fwd)==len(flows_bwd)==len(flows_fwd))
    if type(cam_flows_bwd) not in [tuple, list]:#如果都是单个不是list
        return one_scale(cam_flows_fwd, cam_flows_bwd, flows_fwd, flows_bwd, tgt_img, ref_img_fwd, ref_img_bwd, ws)
    else:
        ret = [one_scale(cam_flows_fwd_, cam_flows_bwd_, flows_fwd_, flows_bwd_, tgt_img, ref_img_fwd, ref_img_bwd, ws)
               for (cam_flows_fwd_, cam_flows_bwd_, flows_fwd_, flows_bwd_) in
               zip(cam_flows_fwd, cam_flows_bwd, flows_fwd, flows_bwd)]

        ws = ws / 2.3
        return ret

#by train;validate called
def compute_joint_mask_for_depth(explainability_mask, rigidity_mask_bwd, rigidity_mask_fwd, THRESH):
    joint_masks = []
    for i in range(len(explainability_mask)):
        exp_mask_one_scale = explainability_mask[i]
        rigidity_mask_fwd_one_scale = (rigidity_mask_fwd[i] > THRESH).type_as(exp_mask_one_scale)
        rigidity_mask_bwd_one_scale = (rigidity_mask_bwd[i] > THRESH).type_as(exp_mask_one_scale)
        exp_mask_one_scale_joint = 1 - (1-exp_mask_one_scale[:,1])*(1-exp_mask_one_scale[:,2]).unsqueeze(1) > 0.5
        joint_mask_one_scale_fwd = logical_or(rigidity_mask_fwd_one_scale.type_as(exp_mask_one_scale), exp_mask_one_scale_joint.type_as(exp_mask_one_scale))
        joint_mask_one_scale_bwd = logical_or(rigidity_mask_bwd_one_scale.type_as(exp_mask_one_scale), exp_mask_one_scale_joint.type_as(exp_mask_one_scale))
        joint_mask_one_scale_fwd = Variable(joint_mask_one_scale_fwd.data, requires_grad=False)
        joint_mask_one_scale_bwd = Variable(joint_mask_one_scale_bwd.data, requires_grad=False)
        joint_mask_one_scale = torch.cat((joint_mask_one_scale_bwd, joint_mask_one_scale_bwd,
                        joint_mask_one_scale_fwd, joint_mask_one_scale_fwd), dim=1)
        joint_masks.append(joint_mask_one_scale)

    return joint_masks



#photometric_reconstruction_loss using
def depth_occlusion_masks(depth, pose, intrinsics, intrinsics_inv):
    flow_cam = [pose2flow(depth.squeeze(), pose[:,i], intrinsics, intrinsics_inv) for i in range(pose.size(1))]
    masks1, masks2 = occlusion_masks(flow_cam[1], flow_cam[2])
    masks0, masks3 = occlusion_masks(flow_cam[0], flow_cam[3])
    masks = torch.stack((masks0, masks1, masks2, masks3), dim=1)
    return masks


def occlusion_masks(flow_bw, flow_fw):
    mag_sq = flow_fw.pow(2).sum(dim=1) + flow_bw.pow(2).sum(dim=1)
    #flow_bw_warped = flow_warp(flow_bw, flow_fw)
    #flow_fw_warped = flow_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw
    flow_diff_bw = flow_bw + flow_fw
    occ_thresh =  0.08 * mag_sq + 1.0
    occ_fw = flow_diff_fw.sum(dim=1) > occ_thresh
    occ_bw = flow_diff_bw.sum(dim=1) > occ_thresh
    return occ_bw.type_as(flow_bw), occ_fw.type_as(flow_fw)
#    return torch.stack((occ_bw.type_as(flow_bw), occ_fw.type_as(flow_fw)), dim=1)


def edge_aware_smoothness_per_pixel(img, pred):
    def gradient_x(img):
      gx = img[:,:,:-1,:] - img[:,:,1:,:]
      return gx

    def gradient_y(img):
      gy = img[:,:,:,:-1] - img[:,:,:,1:]
      return gy

    pred_gradients_x = gradient_x(pred)
    pred_gradients_y = gradient_y(pred)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    smoothness_x = torch.abs(pred_gradients_x) * weights_x
    smoothness_y = torch.abs(pred_gradients_y) * weights_y
    import ipdb; ipdb.set_trace()
    return smoothness_x + smoothness_y






