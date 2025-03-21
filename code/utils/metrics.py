import torch

count = 0

def eval_depth(pred, target):
    assert pred.shape == target.shape
    
    if pred.shape == torch.Size([0]):
        return {'d1': 1.0, 'd2': 1.0, 'd3': 1.0, 'abs_rel': 0.0,
            'sq_rel': 0.0, 'abs': 0.0, 'abs_cent': 0.0, 'rmse': 0.0, 'rmse_log': 0.0, 
            'log10':0.0, 'silog':0.0}

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_with_min = (pred - pred[pred > 0.001].min()) - (target - target[target > 0.001].min())

    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    abs_ = torch.mean(torch.abs(diff))
    abs_centered = torch.mean(torch.abs(diff_with_min))

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': 1.0 if torch.isnan(d1).item() else d1.item(), 
        'd2': 1.0 if torch.isnan(d2).item() else d2.item(), 
        'd3': 1.0 if torch.isnan(d3).item() else d3.item(), 
        'abs_rel': 0.0 if torch.isnan(abs_rel).item() else abs_rel.item(),
        'sq_rel': 0.0 if torch.isnan(sq_rel).item() else sq_rel.item(),
        'abs': 0.0 if torch.isnan(abs_).item() else abs_.item(),
        'abs_cent': 0.0 if torch.isnan(abs_centered).item() else abs_centered.item(),
        'rmse': 0.0 if torch.isnan(rmse).item() else rmse.item(), 
        'rmse_log': 0.0 if torch.isnan(rmse_log).item() else rmse_log.item(), 
        'log10': 0.0 if torch.isnan(log10).item() else log10.item(), 
        'silog': 0.0 if torch.isnan(silog).item() else silog.item()}


def cropping_img(args, pred, gt_depth):
    min_depth_eval = args.min_depth_eval

    max_depth_eval = args.max_depth_eval
    
    pred[torch.isinf(pred)] = max_depth_eval
    pred[torch.isnan(pred)] = min_depth_eval

    valid_mask = torch.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    if args.dataset == 'kitti':
        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            gt_depth = gt_depth[top_margin:top_margin +
                            352, left_margin:left_margin + 1216]            

        if args.kitti_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = torch.zeros(valid_mask.shape).to(
                device=valid_mask.device)

            if args.kitti_crop == 'garg_crop':
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.kitti_crop == 'eigen_crop':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                eval_mask = valid_mask

    elif (args.dataset == 'nyudepthv2') or (args.dataset == 'shapenetsem') or (args.dataset == 'shapenetsem_normalized'):
        eval_mask = torch.zeros(valid_mask.shape).to(device=valid_mask.device)
        eval_mask[45:471, 41:601] = 1
    else:
        eval_mask = valid_mask

    valid_mask = torch.logical_and(valid_mask, eval_mask)

    return pred[valid_mask], gt_depth[valid_mask]