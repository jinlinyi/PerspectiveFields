# Reference: https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS
import torch


def one_scale_gradient_loss(pred_scale, gt, mask):
    mask_float = mask.to(dtype=pred_scale.dtype, device=pred_scale.device)

    d_diff = pred_scale - gt

    v_mask = torch.mul(mask_float[:, :, :-2, :], mask_float[:, :, 2:, :])
    v_gradient = torch.abs(d_diff[:, :, :-2, :] - d_diff[:, :, 2:, :])
    # v_gradient = torch.mul(v_gradient, v_mask)
    v_gradient = v_gradient[v_mask.to(dtype=mask.dtype)]

    h_gradient = torch.abs(d_diff[:, :, :, :-2] - d_diff[:, :, :, 2:])
    h_mask = torch.mul(mask_float[:, :, :, :-2], mask_float[:, :, :, 2:])
    # h_gradient = torch.mul(h_gradient, h_mask)
    h_gradient = h_gradient[h_mask.to(dtype=mask.dtype)]

    valid_num = torch.sum(h_mask) + torch.sum(v_mask)

    gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
    gradient_loss = gradient_loss / (valid_num + 1e-8)
    return gradient_loss


def msgil_norm_loss(pred, gt, valid_mask, scales_num=4):
    """
    GT normalized Multi-scale Gradient Loss Fuction.
    """
    grad_term = 0.0
    # gt_mean = minmax_meanstd[:, 2]
    # gt_std = minmax_meanstd[:, 3]
    gt_trans = (
        gt  # (gt - gt_mean[:, None, None, None]) / (gt_std[:, None, None, None] + 1e-8)
    )
    for i in range(scales_num):
        step = pow(2, i)
        d_gt = gt_trans[:, :, ::step, ::step]
        d_pred = pred[:, :, ::step, ::step]
        d_mask = valid_mask[:, :, ::step, ::step]
        grad_term += one_scale_gradient_loss(d_pred, d_gt, d_mask)
    return grad_term


def meanstd_tanh_norm_loss(pred, gt, mask):
    """
    loss = MAE((d-u)/s - d') + MAE(tanh(0.01*(d-u)/s) - tanh(0.01*d'))
    """
    mask_sum = torch.sum(mask, dim=(1, 2, 3))
    # mask invalid batches
    mask_batch = mask_sum > 100
    if True not in mask_batch:
        return torch.tensor(0.0, dtype=torch.float).cuda()
    mask_maskbatch = mask[mask_batch]
    pred_maskbatch = pred[mask_batch]
    gt = gt[mask_batch]

    B, C, H, W = gt.shape
    loss = 0
    loss_tanh = 0
    for i in range(B):
        mask_i = mask_maskbatch[i, ...]
        pred_depth_i = pred_maskbatch[i, ...][mask_i]
        gt_i = gt[i, ...][mask_i]

        depth_diff = torch.abs(gt_i - pred_depth_i)
        loss += torch.mean(depth_diff)

        tanh_norm_gt = torch.tanh(0.01 * gt_i)
        tanh_norm_pred = torch.tanh(0.01 * pred_depth_i)
        loss_tanh += torch.mean(torch.abs(tanh_norm_gt - tanh_norm_pred))
    loss_out = loss / B + loss_tanh / B
    return loss_out.float()
