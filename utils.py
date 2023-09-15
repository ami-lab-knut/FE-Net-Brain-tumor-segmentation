import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, average_precision_score
from hausdorff import hd95

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve


def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    # print(true.get_device())
    # print(logits.get_device())
    # print('logits type: ', logits.dtype)
    # print(logits)
    
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        # print(num_classes, ' ',  true.shape)
        # print('eye shape: ', torch.eye(num_classes).shape)
        true_1_hot = torch.eye(num_classes)[true.squeeze(1).cpu()]
        # print('after---: ', true_1_hot.shape)
        true_1_hot = true_1_hot.permute(0, 4, 1, 2,3).float()
        # print('one hot gt shape: ', true_1_hot.shape)
        # print('predicted shape: ', logits.shape)
        probas = F.softmax(logits, dim=1)
        # print('probabs: ', torch.max(probas), probas)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    # print('uniques: ', torch.unique(probas), torch.unique(true_1_hot))
    intersection = torch.sum(probas * true_1_hot, dims)
    # print('intersection: ', intersection)
    cardinality = torch.sum(probas + true_1_hot, dims)
    # print('union: ', cardinality)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1).cpu()]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2,3).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

def dice_loss_2d(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    # print(true.get_device())
    # print(logits.get_device())
    # print('logits type: ', logits.dtype)
    # print(logits)
    
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        # print(num_classes, ' ',  true.shape)
        # print('eye shape: ', torch.eye(num_classes).shape)
        true_1_hot = torch.eye(num_classes)[true.squeeze(1).cpu()]
        # print('after---: ', true_1_hot.shape)
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        
        probas = F.softmax(logits, dim=1)
        # print('probabs: ', torch.max(probas), probas)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    # print('uniques: ', torch.unique(probas), torch.unique(true_1_hot))
    intersection = torch.sum(probas * true_1_hot, dims)
    # print('intersection: ', intersection)
    cardinality = torch.sum(probas + true_1_hot, dims)
    # print('union: ', cardinality)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def jaccard_loss_2d(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1).cpu()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

def tversky_loss(true, logits, alpha, beta, eps=1e-7):
    """Computes the Tversky loss [1].
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes).cuda()
        true_1_hot = true_1_hot[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)

def dice_coeff(pred, target):
    target = target.contiguous()
    # print('pred and target shapes: ', pred.shape, ' ', target.shape)
    smooth = 0.001
    #print(pred.shape, pred.shape[0])
    #print('--size: ', torch.Tensor(pred).size(0))
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    #print('reshaped shapes: ', m1.shape, ' ', m2.shape)
    intersection = (m1 * m2).sum()
    
    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    iou = (intersection + smooth) / (m1.sum() + m2.sum() - intersection + smooth)
    return dice, iou


def class_dice(pred_class, target, tot_classes, epsilon = 1e-6):
    num_classes = torch.unique(target)
    #print('num classes: ', num_classes)
    #pred_class = torch.argmax(pred, dim = 1)
    #pred_class = torch.argmax(pred.squeeze(), dim=1).detach().cpu().numpy()
    dice = torch.zeros(3)
    dscore = torch.zeros(3)
    iou_score = torch.zeros(3)
    ds = 0
    ious = 0
    for c in range(1, tot_classes):
        if c in num_classes:
            #print('c: ', c)
            p = (pred_class == c)
            t = (target == c)
            #print('p shape: ', p.shape)
            #print('t shape: ', t.shape)
            dc, iou = dice_coeff(p, t)
            #print('dc done')
            dice[c-1] = 1 - dc
            dscore[c-1] = dc
            iou_score[c-1] = iou
            #print('appended')
            dl = torch.sum(dice)
            ds = torch.mean(dscore)
            ious = torch.mean(iou_score)
        
    return ds, dscore, ious, iou_score

def weights(pred, target, epsilon = 1e-6):
    num_classes = 4
    pred_class = torch.argmax(pred, dim = 1)
    #pred_class = torch.argmax(pred.squeeze(), dim=1).detach().cpu().numpy()
    dice = np.ones(num_classes)
    tot = 0
    for c in range(num_classes):
        t = (target == c).sum()
        tot = tot + t
        #print(t.shape)
        dice[c] = t

    dice = dice/dice.sum()
    dice = 1 - dice
    
    return torch.from_numpy(dice).float()

def dice_lossss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

from monai.losses.dice import DiceLoss, one_hot

def get_weights(target):
    sum0 = torch.sum(target[target == 0])
    sum1 = torch.sum(target[target == 1])
    sum2 = torch.sum(target[target == 2])
    sum3 = torch.sum(target[target == 3])
    total = sum0 + sum1 + sum2 + sum3
    
    weight_tensor = torch.Tensor([sum0/total, sum1/total, sum2/total, sum3/total])
    return weight_tensor

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.3, gamma=0.7, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

def loss_segmentation_hem(pred, target):
    target = torch.squeeze(target, dim = 1)
    # print('pred/ target shapes: ', pred.shape, target.shape)
    lossf = nn.CrossEntropyLoss()#get_weights(target).cuda())
    ce = lossf(pred, target)
    # print('target unique: ', torch.unique(target))
    arg = torch.argmax(pred, dim = 1)
    # print('pred arg unique: ', torch.unique(arg))
    
    mask_wt = torch.where((arg == 3) | (arg == 2) | (arg == 1), 1, 0)
    mask_tc = torch.where((arg == 3) | (arg == 1), 1, 0)
    # print('mask shapes: ', mask_wt.shape, mask_tc.shape)
    
    dsc = dice_loss(target, pred)
    ious = jaccard_loss(target, pred)
    
    # target = torch.squeeze(target, dim = 0)
    # print('before tversky shapes: ', target.shape, pred.shape)
    tl1 = tversky_loss(target, pred, alpha = 0.3, beta = 0.7)
    tl2 = tversky_loss(target, pred, alpha = 0.7, beta = 0.3)
    
    loss = (dsc + ious + ce)/3# + (tl1 + tl2)/2

    return loss

def loss_segmentation(pred, target):
    #print('pred/target shape: ', pred.shape, target.shape)
    #print('pred/target max; ', torch.max(pred), torch.max(target), torch.unique(target))
    
    # lossf = nn.CrossEntropyLoss(weight = weights(pred, target).cuda())
    #w = np.array([0.001503597, 0.5840496, 0.20685249, 1])
    #lossf = nn.CrossEntropyLoss(weight = torch.from_numpy(w).float().cuda())
    # ce = lossf(pred, target)
    
    # print('target/pred dtypes: ', target.dtype, pred.dtype)
    dsc_l = dice_loss_2d(target, pred)
    iou_l = jaccard_loss_2d(target, pred)
    # tl = tversky_loss(target, pred, alpha = 0.7, beta = 0.3)
    
    
    pred_et = pred.clone()
    target_et = target.clone()
    pred_et_0 = torch.unsqueeze(pred_et[:, 0, :, :], dim = 1)
    pred_et_3 = torch.unsqueeze(pred_et[:, 3, :, :], dim = 1)
    pred_et_only = torch.cat((pred_et_0, pred_et_3), dim = 1)
    # print('pred channels only: ', pred_et_only.shape)
    # print('pred/target shape: ', pred_et.shape, target_et.shape)

    target_et[target_et < 3] = 0
    target_et[target_et == 3] = 1
    
    # et_dice = dice_loss(target_et, pred_et_only)
    # et_iou = jaccard_loss(target_et, pred_et_only)
    
    pred = torch.argmax(pred, dim = 1)
    #print('pred shape: ', pred.shape)
    #print('pred unique: ', torch.unique(pred))
    dsc, class_dsc, ious, class_iou = class_dice(pred, target, 4)
    
    
    pred_tc = pred.clone()
    target_tc = target.clone()
    pred_tc[pred_tc == 3] = 1
    pred_tc[pred_tc == 2] = 0
    target_tc[target_tc == 3] = 1
    target_tc[target_tc == 2] = 0
    #print('tc unique: ', torch.unique(target_tc))
    tc_dice, tc_iou = dice_coeff(pred_tc, target_tc)
    #tc_dice, _, tc_iou, _ = class_dice(pred_tc, target_tc, 2)
    
    pred_whole = pred.clone()
    target_whole = target.clone()
    pred_whole[pred_whole > 1] = 1
    target_whole[target_whole > 1] = 1
    #print('whole unique: ', torch.unique(target_whole))
    whole_dice, whole_iou = dice_coeff(pred_whole, target_whole)
    #whole_dice, _, whole_iou, _ = class_dice(pred_whole, target_whole, 2)
    #print((et_dice))
    
    loss = (dsc_l + iou_l)# + tl**(2)# + (1 - et_dice) + (1 - et_iou)
    #loss = tl
    
    return loss, dsc, class_dsc, ious, class_iou, [tc_dice, tc_iou], [whole_dice, whole_iou]


def loss_detection(pred, target):
    lossf = nn.CrossEntropyLoss()
    
    ce = lossf(pred, target)
    
    return ce

def prec_rec(pred, gt):
    tn, fp, fn, tp = confusion_matrix(pred.ravel(), gt.ravel()).ravel()
    
    prec = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    specificity = (tn)/(tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    
    iou = (tp)/(tp + fp + fn)
    
    return prec, recall, specificity, accuracy, iou, fp, fn

def other_metrics(pred, target):
    
    pred_et = pred.clone()
    target_et = target.clone()
    pred_et[pred_et != 3] = 0
    pred_et[pred_et == 3] = 1
    target_et[target_et != 3] = 0
    target_et[target_et == 3] = 1
    prec_et, recall_et, specificity_et, acc_et, _, fp_et, fn_et = prec_rec(pred_et.detach().cpu().numpy(), target_et.detach().cpu().numpy())
    
    pred_tc = pred.clone()
    target_tc = target.clone()
    pred_tc[pred_tc == 3] = 1
    pred_tc[pred_tc == 2] = 0
    target_tc[target_tc == 3] = 1
    target_tc[target_tc == 2] = 0
    prec_tc, recall_tc, specificity_tc, acc_tc, _, fp_tc, fn_tc = prec_rec(pred_tc.detach().cpu().numpy(), target_tc.detach().cpu().numpy())
    
    pred_whole = pred.clone()
    target_whole = target.clone()
    pred_whole[pred_whole > 1] = 1
    target_whole[target_whole > 1] = 1
    prec_whole, recall_whole, specificity_whole, acc_whole, _, fp_whole, fn_whole = prec_rec(pred_whole.detach().cpu().numpy(), target_whole.detach().cpu().numpy())
    
    return [recall_et, recall_tc, recall_whole], [specificity_et, specificity_tc, specificity_whole], [fp_et, fp_tc, fp_whole], [fn_et, fn_tc, fn_whole]

def test_scores_3d(pred, target):
    
    dsc, class_dsc, ious, class_iou = class_dice(pred, target, 4)
    
    pred_tc = pred.clone()
    target_tc = target.clone()
    pred_tc[pred_tc == 3] = 1
    pred_tc[pred_tc == 2] = 0
    target_tc[target_tc == 3] = 1
    target_tc[target_tc == 2] = 0
    
    tc_dice, tc_iou = dice_coeff(pred_tc, target_tc)
    #tc_dice, _, tc_iou, _ = class_dice(pred_tc, target_tc, 2)
    
    pred_whole = pred.clone()
    target_whole = target.clone()
    pred_whole[pred_whole > 1] = 1
    target_whole[target_whole > 1] = 1
    #print('whole unique: ', torch.unique(target_whole))
    whole_dice, whole_iou = dice_coeff(pred_whole, target_whole)
    #whole_dice, _, whole_iou, _ = class_dice(pred_whole, target_whole, 2)
    
    return dsc, class_dsc, ious, class_iou, [tc_dice, tc_iou], [whole_dice, whole_iou]


class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
            return np.array([np.Inf])

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.max(distances[indexes]))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
        ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()

        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
        ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
        ).float()

        return torch.max(right_hd, left_hd)

def calculate_hd95_multi_class(preds, target, spacing=None, connectivity=1):
    hd95_dict = {}
    hd95_dict['mean'] = 0.0
    hd95_dict['N-NE'] = 0.0
    hd95_dict['ED'] = 0.0
    hd95_dict['ET'] = 0.0

    target = F.one_hot(target, 4).permute(0,4,1,2,3).float()
    preds = F.one_hot(preds.argmax(1), 4).permute(0,4,1,2,3).float()

    preds = preds[:, 1:, :, :, :]
    target = target[:, 1:, :, :, :]

    assert preds.size() == target.size()

    for i in range(preds.shape[0]):
        batch_hd95_dict = hd95(preds[i, ...], target[i, ...], spacing, connectivity)
        hd95_dict['mean'] += batch_hd95_dict['mean']
        hd95_dict['N-NE'] += batch_hd95_dict['N-NE']
        hd95_dict['ED'] += batch_hd95_dict['ED']
        hd95_dict['ET'] += batch_hd95_dict['ET']

    hd95_dict['mean'] /= preds.shape[0]
    hd95_dict['N-NE'] /= preds.shape[0]
    hd95_dict['ED'] /= preds.shape[0]
    hd95_dict['ET'] /= preds.shape[0]

    return hd95_dict

from medpy.metric import hd95
import os

def hausdorf_distance(pred, target):
    # print('pred/target shapes: ', pred.shape, target.shape)
    hd95_dict = {}
    hd95_dict['mean'] = 0.0
    hd95_dict['ET'] = 0.0
    hd95_dict['TC'] = 0.0
    hd95_dict['WT'] = 0.0
    
    pred_et = pred.clone()
    target_et = target.clone()
    pred_et[pred_et != 3] = 0
    pred_et[pred_et == 3] = 1
    target_et[target_et != 3] = 0
    target_et[target_et == 3] = 1
    
    # HD = HausdorffDistance()
    
    # print('et uniques: ', torch.unique(target_et), torch.unique(pred_et))
    
    if len(torch.unique(pred_et)) >= 2 and len(torch.unique(target_et)) >= 2:
        hd95_dict['ET'] = hd95(pred_et.detach().cpu().numpy(), target_et.detach().cpu().numpy())
        # hd95_dict['ET'] = HD.compute(pred_et, target_et)
        # print('calculated: ', hd95_dict['ET'])
    elif len(torch.unique(pred_et)) == 1 and len(torch.unique(target_et)) == 1:
        hd95_dict['ET'] = 0
    else:
        # print('ET uniques: ', torch.unique(pred_et), torch.unique(target_et))
        hd95_dict['ET'] = 80
        pth = 'D:\\brain_tumor_segmentation\\rough_4_modified_3d\\special_HD_saves'
        np.save(os.path.join(pth, 'pred.npy'), pred_et.detach().cpu().numpy())
        np.save(os.path.join(pth, 'target.npy'), target_et.detach().cpu().numpy())
    
    pred_tc = pred.clone()
    target_tc = target.clone()
    pred_tc[pred_tc == 3] = 1
    pred_tc[pred_tc == 2] = 0
    target_tc[target_tc == 3] = 1
    target_tc[target_tc == 2] = 0
    
    if len(torch.unique(pred_tc)) >= 2 and len(torch.unique(target_tc)) >= 2:
        hd95_dict['TC'] = hd95(pred_tc.detach().cpu().numpy(), target_tc.detach().cpu().numpy())
        # hd95_dict['TC'] = HD.compute(pred_tc, target_tc)
    else:
        # print('TC uniques: ', torch.unique(pred_tc), torch.unique(target_tc))
        hd95_dict['TC'] = 80
    
    pred_whole = pred.clone()
    target_whole = target.clone()
    pred_whole[pred_whole > 1] = 1
    target_whole[target_whole > 1] = 1
    
    if len(torch.unique(pred_whole)) >= 2 and len(torch.unique(target_whole)) >= 2:
        hd95_dict['WT'] = hd95(pred_whole.detach().cpu().numpy(), target_whole.detach().cpu().numpy())
        # hd95_dict['WT'] = HD.compute(pred_whole, target_whole)
    else:
        # print('WT uniques: ', torch.unique(pred_whole), torch.unique(target_whole))
        hd95_dict['WT'] = 80
    
    return hd95_dict

def dice_3d(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:

        true_1_hot = torch.eye(num_classes)[true.squeeze(1).cpu()]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2,3).float()

        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return dice_loss


