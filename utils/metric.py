import numpy as np
import torch

# https://github.com/davisvideochallenge/davis/blob/master/python/lib/davis/measures/jaccard.py
def eval_iou(annotation, segmentation):
    """ Compute region similarity as the Jaccard Index.

    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.

    Return:
        jaccard (float): region similarity

    """
    annotation = annotation.astype(np.bool_)
    segmentation = segmentation.astype(np.bool_)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
               np.sum((annotation | segmentation), dtype=np.float32)


# https://github.com/svip-lab/PlanarReconstruction/blob/master/utils/metric.py
# https://github.com/art-programmer/PlaneNet/blob/master/utils.py#L2115
def eval_plane_recall_depth(predSegmentations, gtSegmentations, predDepths, gtDepths, pred_plane_num, threshold=0.5):
    predNumPlanes = pred_plane_num  # actually, it is the maximum number of the predicted planes

    if 20 in np.unique(gtSegmentations):  # in GT plane Seg., number '20' indicates non-plane
        gtNumPlanes = len(np.unique(gtSegmentations)) - 1
    else:
        gtNumPlanes = len(np.unique(gtSegmentations))

    if len(gtSegmentations.shape) == 2:
        gtSegmentations = (np.expand_dims(gtSegmentations, -1) == np.arange(gtNumPlanes)).astype(np.float32)  # h, w, gtNumPlanes
    if len(predSegmentations.shape) == 2:
        predSegmentations = (np.expand_dims(predSegmentations, -1) == np.arange(predNumPlanes)).astype(np.float32)  # h, w, predNumPlanes

    # predSegmentations_ = torch.from_numpy(predSegmentations).cuda()  # h, w
    # gtSegmentations_ = torch.from_numpy(gtSegmentations).cuda()  # h, w
    # # predict
    # pred_masks = []
    # for i in range(predNumPlanes):
    #     mask_i = predSegmentations_ == i
    #     mask_i = mask_i.float()
    #     if mask_i.sum() > 0:
    #         pred_masks.append(mask_i)
    # masks_pred = torch.stack(pred_masks, dim=-1)
    # predSegmentations = torch.round(masks_pred).cpu().numpy()  # h, w, N
    #
    # # gt
    # gt_masks = []
    # for i in range(20):
    #     mask_i = gtSegmentations_ == i
    #     mask_i = mask_i.float()
    #     if mask_i.sum() > 0:
    #         gt_masks.append(mask_i)
    # gtSegmentations = torch.stack(gt_masks, dim=-1).cpu().numpy()
    # gtNumPlanes = gtSegmentations.shape[-1]


    planeAreas = gtSegmentations.sum(axis=(0, 1))  # gt plane pixel number

    intersectionMask = np.expand_dims(gtSegmentations, -1) * np.expand_dims(predSegmentations, 2) > 0.5  # h, w, gtNumPlanes, predNumPlanes

    depthDiffs = gtDepths - predDepths  # h, w
    depthDiffs = depthDiffs[:, :, np.newaxis, np.newaxis]  # h, w, 1, 1

    intersection = np.sum((intersectionMask).astype(np.float32), axis=(0, 1))  # gtNumPlanes, predNumPlanes

    planeDiffs = np.abs(depthDiffs * intersectionMask).sum(axis=(0, 1)) / np.maximum(intersection, 1e-4)  # gtNumPlanes, predNumPlanes
    planeDiffs[intersection < 1e-4] = 1

    union = np.sum(
        ((np.expand_dims(gtSegmentations, -1) + np.expand_dims(predSegmentations, 2)) > 0.5).astype(np.float32),
        axis=(0, 1))  # gtNumPlanes, predNumPlanes
    planeIOUs = intersection / np.maximum(union, 1e-4)  # gtNumPlanes, predNumPlanes

    numPredictions = int(predSegmentations.max(axis=(0, 1)).sum())

    numPixels = planeAreas.sum()
    IOUMask = (planeIOUs > threshold).astype(np.float32)

    minDiff = np.min(planeDiffs * IOUMask + 1000000 * (1 - IOUMask), axis=1)

    stride = 0.05
    pixelRecalls = []
    planeStatistics = []
    for step in range(int(0.61 / stride + 1)):
        diff = step * stride
        pixelRecalls.append(np.minimum((intersection * (planeDiffs <= diff).astype(np.float32) * IOUMask).sum(1),
                                       planeAreas).sum() / numPixels)
        planeStatistics.append(((minDiff <= diff).sum(), gtNumPlanes, numPredictions))
    return pixelRecalls, planeStatistics

# https://github.com/svip-lab/PlanarReconstruction/blob/master/utils/metric.py
def eval_plane_recall_normal(segmentation, gt_segmentation, param, gt_param, pred_non_plane_idx, threshold=0.5):
    """
    :param segmentation: label map for plane segmentation [h, w] where 20 indicate non-planar
    :param gt_segmentation: ground truth label for plane segmentation where 20 indicate non-planar
    :param threshold: value for iou
    :return: percentage of correctly predicted ground truth planes correct plane
    """
    ''''''
    pred_plane_idxs = np.unique(segmentation)
    if pred_non_plane_idx in pred_plane_idxs:
        pred_plane_idx_max = pred_plane_idxs[-2]
    else:
        pred_plane_idx_max = pred_plane_idxs[-1]
    plane_num = pred_plane_idx_max + 1

    if 20 in np.unique(gt_segmentation):  # in GT plane Seg., number '20' indicates non-plane
        gt_plane_num = len(np.unique(gt_segmentation)) - 1
    else:
        gt_plane_num = len(np.unique(gt_segmentation))
    ''''''
    '''
    plane_num = pred_non_plane_idx - 1
    # gt
    gt_plane_num = 0
    for i in range(20):
        mask_i = gt_segmentation == i
        mask_i = mask_i
        if mask_i.sum() > 0:
            gt_plane_num += 1

    gt_idxs = np.unique(gt_segmentation)
    assert gt_idxs[gt_plane_num - 1] == gt_plane_num - 1
    '''

    # 13: 0:0.05:0.6
    depth_threshold_list = np.linspace(0.0, 30, 13)
    plane_recall = np.zeros((gt_plane_num, len(depth_threshold_list)))
    pixel_recall = np.zeros((gt_plane_num, len(depth_threshold_list)))

    plane_area = 0.0

    gt_param = gt_param.reshape(20, 3)

    # check if plane is correctly predict
    for i in range(gt_plane_num):
        gt_plane = gt_segmentation == i
        plane_area += np.sum(gt_plane)

        for j in range(plane_num):
            pred_plane = segmentation == j
            iou = eval_iou(gt_plane, pred_plane)

            if iou > threshold:
                # mean degree difference over overlap region:
                gt_p = gt_param[i]
                pred_p = param[j]

                n_gt_p = gt_p / np.linalg.norm(gt_p)
                n_pred_p = pred_p / np.linalg.norm(pred_p)

                angle = np.arccos(np.clip(np.dot(n_gt_p, n_pred_p), -1.0, 1.0))
                degree = np.degrees(angle)
                depth_diff = degree

                # compare with threshold difference
                plane_recall[i] = (depth_diff < depth_threshold_list).astype(np.float32)
                pixel_recall[i] = (depth_diff < depth_threshold_list).astype(np.float32) * \
                                  (np.sum(gt_plane * pred_plane))
                break

    pixel_recall = np.sum(pixel_recall, axis=0).reshape(-1) / plane_area

    plane_recall_new = np.zeros((len(depth_threshold_list), 3))
    plane_recall = np.sum(plane_recall, axis=0).reshape(-1, 1)
    plane_recall_new[:, 0:1] = plane_recall
    plane_recall_new[:, 1] = gt_plane_num
    plane_recall_new[:, 2] = plane_num

    return plane_recall_new, pixel_recall

# https://github.com/svip-lab/PlanarReconstruction/blob/master/utils/metric.py
# https://github.com/yi-ming-qian/interplane/blob/master/utils/metric.py
def evaluateMasks(predSegmentations, gtSegmentations, device, pred_non_plane_idx, gt_non_plane_idx=20, printInfo=False):
    """
    :param predSegmentations:
    :param gtSegmentations:
    :param device:
    :param pred_non_plane_idx:
    :param gt_non_plane_idx:
    :param printInfo:
    :return:
    """
    predSegmentations = torch.from_numpy(predSegmentations).to(device)
    gtSegmentations = torch.from_numpy(gtSegmentations).to(device)

    pred_masks = []
    if pred_non_plane_idx > 0:
        for i in range(pred_non_plane_idx):
            mask_i = predSegmentations == i
            mask_i = mask_i.float()
            if mask_i.sum() > 0:
                pred_masks.append(mask_i)
    else:
        assert pred_non_plane_idx == -1 or pred_non_plane_idx == 0
        for i in range(gt_non_plane_idx + 1, 100):
            mask_i = predSegmentations == i
            mask_i = mask_i.float()
            if mask_i.sum() > 0:
                pred_masks.append(mask_i)
    predMasks = torch.stack(pred_masks, dim=0)

    gt_masks = []
    if gt_non_plane_idx > 0:
        for i in range(gt_non_plane_idx):
            mask_i = gtSegmentations == i
            mask_i = mask_i.float()
            if mask_i.sum() > 0:
                gt_masks.append(mask_i)
    else:
        assert pred_non_plane_idx == -1 or pred_non_plane_idx == 0
        for i in range(gt_non_plane_idx+1, 100):
            mask_i = gtSegmentations == i
            mask_i = mask_i.float()
            if mask_i.sum() > 0:
                gt_masks.append(mask_i)
    gtMasks = torch.stack(gt_masks, dim=0)

    valid_mask = (gtMasks.max(0)[0]).unsqueeze(0)

    gtMasks = torch.cat([gtMasks, torch.clamp(1 - gtMasks.sum(0, keepdim=True), min=0)], dim=0)  # M+1, H, W
    predMasks = torch.cat([predMasks, torch.clamp(1 - predMasks.sum(0, keepdim=True), min=0)], dim=0)  # N+1, H, W

    intersection = (gtMasks.unsqueeze(1) * predMasks * valid_mask).sum(-1).sum(-1).float()
    union = (torch.max(gtMasks.unsqueeze(1), predMasks) * valid_mask).sum(-1).sum(-1).float()

    N = intersection.sum()

    RI = 1 - ((intersection.sum(0).pow(2).sum() + intersection.sum(1).pow(2).sum()) / 2 - intersection.pow(2).sum()) / (
            N * (N - 1) / 2)
    joint = intersection / N
    marginal_2 = joint.sum(0)
    marginal_1 = joint.sum(1)
    H_1 = (-marginal_1 * torch.log2(marginal_1 + (marginal_1 == 0).float())).sum()
    H_2 = (-marginal_2 * torch.log2(marginal_2 + (marginal_2 == 0).float())).sum()

    B = (marginal_1.unsqueeze(-1) * marginal_2)
    log2_quotient = torch.log2(torch.clamp(joint, 1e-8) / torch.clamp(B, 1e-8)) * (torch.min(joint, B) > 1e-8).float()
    MI = (joint * log2_quotient).sum()
    voi = H_1 + H_2 - 2 * MI

    IOU = intersection / torch.clamp(union, min=1)
    SC = ((IOU.max(-1)[0] * torch.clamp((gtMasks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N + (
            IOU.max(0)[0] * torch.clamp((predMasks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N) / 2
    info = [RI.item(), voi.item(), SC.item()]
    if printInfo:
        print('mask statistics', info)
        pass
    return info


def evaluateDepths(predDepths, gtDepths, printInfo=False):
    """Evaluate depth reconstruction accuracy"""
    masks = gtDepths > 1e-4

    numPixels = float(masks.sum())

    rmse = np.sqrt((pow(predDepths - gtDepths, 2) * masks).sum() / numPixels)
    rmse_log = np.sqrt(
        (pow(np.log(np.maximum(predDepths, 1e-4)) - np.log(np.maximum(gtDepths, 1e-4)), 2) * masks).sum() / numPixels)
    log10 = (np.abs(
        np.log10(np.maximum(predDepths, 1e-4)) - np.log10(np.maximum(gtDepths, 1e-4))) * masks).sum() / numPixels
    rel = (np.abs(predDepths - gtDepths) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    rel_sqr = (pow(predDepths - gtDepths, 2) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    deltas = np.maximum(predDepths / np.maximum(gtDepths, 1e-4), gtDepths / np.maximum(predDepths, 1e-4)) + (
                1 - masks.astype(np.float32)) * 10000
    accuracy_1 = (deltas < 1.25).sum() / numPixels
    accuracy_2 = (deltas < pow(1.25, 2)).sum() / numPixels
    accuracy_3 = (deltas < pow(1.25, 3)).sum() / numPixels
    if printInfo:
        print(('depth statistics', rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3))
        pass
    return [rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3]