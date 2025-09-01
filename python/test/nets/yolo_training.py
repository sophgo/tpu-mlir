import math
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn


class YOLOLoss(nn.Module):

    def __init__(self,
                 anchors,
                 num_classes,
                 input_shape,
                 cuda,
                 anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 label_smoothing=0):
        super(YOLOLoss, self).__init__()
        #-----------------------------------------------------------#
        # The anchors for the 20x20 feature layer are [116,90], [156,198], [373,326].
        # The 40x40 feature layer corresponds to anchors [30,61], [62,45], [59,119].
        # The anchor for the 80x80 feature layer is [10,13], [16,30], [33,23]
        #-----------------------------------------------------------#
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing

        self.threshold = 4

        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 1 * (input_shape[0] * input_shape[1]) / (640**2)
        self.cls_ratio = 0.5 * (num_classes / 80)
        self.cuda = cuda

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_giou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        返回为：
        -------
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        #----------------------------------------------------#
        # Calculate the top-left and bottom-right corners of the predicted bounding box.
        #----------------------------------------------------#
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        #----------------------------------------------------#
        # Calculate the top-left and bottom-right corners of the ground truth box.
        #----------------------------------------------------#
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        #----------------------------------------------------#
        # Compute all IoUs between ground truth and predicted boxes
        #----------------------------------------------------#
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins,
                                 torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / union_area

        #----------------------------------------------------#
        # Find the top-left and bottom-right corners of the minimal bounding box enclosing two boxes.
        #----------------------------------------------------#
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        #----------------------------------------------------#
        # Calculate diagonal distance
        #----------------------------------------------------#
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou = iou - (enclose_area - union_area) / enclose_area

        return giou

    #---------------------------------------------------#
    # Label Smoothing
    #---------------------------------------------------#
    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def forward(self, l, input, targets=None, y_true=None):
        #----------------------------------------------------#
        # l represents the nth effective feature layer
        # input's shape is bs, 3*(5+num_classes), 20, 20
        #                   bs, 3*(5+num_classes), 40, 40
        #                   bs, 3*(5+num_classes), 80, 80
        # targets Label information of ground truth boxes [ batch_size, num_gt, 5 ]
        #----------------------------------------------------#
        #--------------------------------#
        # Get image count, feature layer height and width
        #   20, 20
        #--------------------------------#
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        #-----------------------------------------------------------------------#
        # Calculate step length
        # How many pixels correspond to each feature point in the original image?
        # [640, 640] The height stride is 640 / 20 = 32, width stride is 640 / 20 = 32
        # If the feature layer is 20x20, a feature point corresponds to 32 pixels on the original image.
        # If the feature layer is 40x40, a feature point corresponds to 16 pixels in the original image.
        # If the feature layer is 80x80, a feature point corresponds to 8 pixels on the original image.
        #   stride_h = stride_w = 32、16、8
        #-----------------------------------------------------------------------#
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        #-------------------------------------------------#
        # The size of the scaled_anchors obtained here is relative to the feature layer.
        #-------------------------------------------------#
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        #-----------------------------------------------#
        # There are three inputs, their shapes are respectively.
        #   bs, 3 * (5+num_classes), 20, 20 => bs, 3, 5 + num_classes, 20, 20 => batch_size, 3, 20, 20, 5 + num_classes

        #   batch_size, 3, 20, 20, 5 + num_classes
        #   batch_size, 3, 40, 40, 5 + num_classes
        #   batch_size, 3, 80, 80, 5 + num_classes
        #-----------------------------------------------#
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h,
                                in_w).permute(0, 1, 3, 4, 2).contiguous()

        #-----------------------------------------------#
        # Prior box center adjustment parameters
        #-----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        #-----------------------------------------------#
        # Prior box width and height adjustment parameters
        #-----------------------------------------------#
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        #-----------------------------------------------#
        # Get confidence score, object present?
        #-----------------------------------------------#
        conf = torch.sigmoid(prediction[..., 4])
        #-----------------------------------------------#
        # Type Confidence Level
        #-----------------------------------------------#
        pred_cls = torch.sigmoid(prediction[..., 5:])
        #-----------------------------------------------#
        # self.get_target has been merged into the dataloader
        # The reason is that slow execution here will greatly extend the training time.
        #-----------------------------------------------#
        # y_true, noobj_mask = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        #---------------------------------------------------------------#
        # Decode the prediction results and assess their overlap with true values.
        # If the overlap is too large, ignore these feature points as they are predicted to be relatively accurate.
        # Not suitable as a negative sample
        #----------------------------------------------------------------#
        pred_boxes = self.get_pred_boxes(l, x, y, h, w, targets, scaled_anchors, in_h, in_w)

        if self.cuda:
            y_true = y_true.type_as(x)

        loss = 0
        n = torch.sum(y_true[..., 4] == 1)
        if n != 0:
            #---------------------------------------------------------------#
            # Calculate the GIoU between predicted and true results, and compute the GIoU loss for prior boxes with corresponding true boxes.
            # loss_cls computes the classification loss for prior boxes corresponding to real boxes.
            #----------------------------------------------------------------#
            giou = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
            loss_loc = torch.mean((1 - giou)[y_true[..., 4] == 1])
            loss_cls = torch.mean(
                self.BCELoss(
                    pred_cls[y_true[..., 4] == 1],
                    self.smooth_labels(y_true[..., 5:][y_true[..., 4] == 1], self.label_smoothing,
                                       self.num_classes)))
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio
            #-----------------------------------------------------------#
            # Compute confidence loss
            # This also means the corresponding predicted box is more accurate.
            # It is actually used to predict this object.
            #-----------------------------------------------------------#
            tobj = torch.where(y_true[..., 4] == 1,
                               giou.detach().clamp(0), torch.zeros_like(y_true[..., 4]))
        else:
            tobj = torch.zeros_like(y_true[..., 4])
        loss_conf = torch.mean(self.BCELoss(conf, tobj))

        loss += loss_conf * self.balance[l] * self.obj_ratio
        # if n != 0:
        #     print(loss_loc * self.box_ratio, loss_cls * self.cls_ratio, loss_conf * self.balance[l] * self.obj_ratio)
        return loss

    def get_near_points(self, x, y, i, j):
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]

    def get_target(self, l, targets, anchors, in_h, in_w):
        #-----------------------------------------------------#
        # Count total images
        #-----------------------------------------------------#
        bs = len(targets)
        #-----------------------------------------------------#
        # used to select anchor boxes that do not contain objects
        #   bs, 3, 20, 20
        #-----------------------------------------------------#
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        #-----------------------------------------------------#
        # Help to find the corresponding ground truth box for each prior box.
        #-----------------------------------------------------#
        box_best_ratio = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        #-----------------------------------------------------#
        #   batch_size, 3, 20, 20, 5 + num_classes
        #-----------------------------------------------------#
        y_true = torch.zeros(bs,
                             len(self.anchors_mask[l]),
                             in_h,
                             in_w,
                             self.bbox_attrs,
                             requires_grad=False)
        for b in range(bs):
            if len(targets[b]) == 0:
                continue
            batch_target = torch.zeros_like(targets[b])
            #-------------------------------------------------------#
            # Calculate the center point of positive samples on the feature layer.
            # Get the ground truth box size relative to the feature layer.
            #-------------------------------------------------------#
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()

            #-----------------------------------------------------------------------------#
            #   batch_target                                    : num_true_box, 5
            #   batch_target[:, 2:4]                            : num_true_box, 2
            #   torch.unsqueeze(batch_target[:, 2:4], 1)        : num_true_box, 1, 2
            #   anchors                                         : 9, 2
            #   torch.unsqueeze(torch.FloatTensor(anchors), 0)  : 1, 9, 2
            #   ratios_of_gt_anchors    : num_true_box, 9, 2
            #   ratios_of_anchors_gt    : num_true_box, 9, 2
            #
            #   ratios                  : num_true_box, 9, 4
            #   max_ratios              : num_true_box, 9
            # max_ratios is the maximum aspect ratio between each real box and each prior box!
            #------------------------------------------------------------------------------#
            ratios_of_gt_anchors = torch.unsqueeze(batch_target[:, 2:4], 1) / torch.unsqueeze(
                torch.FloatTensor(anchors), 0)
            ratios_of_anchors_gt = torch.unsqueeze(torch.FloatTensor(anchors), 0) / torch.unsqueeze(
                batch_target[:, 2:4], 1)
            ratios = torch.cat([ratios_of_gt_anchors, ratios_of_anchors_gt], dim=-1)
            max_ratios, _ = torch.max(ratios, dim=-1)

            for t, ratio in enumerate(max_ratios):
                #-------------------------------------------------------#
                #   ratio : 9
                #-------------------------------------------------------#
                over_threshold = ratio < self.threshold
                over_threshold[torch.argmin(ratio)] = True
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue
                    #----------------------------------------#
                    # Get the grid cell for the real box
                    #   x  1.25     => 1
                    #   y  3.75     => 3
                    #----------------------------------------#
                    i = torch.floor(batch_target[t, 0]).long()
                    j = torch.floor(batch_target[t, 1]).long()

                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]

                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue

                        if box_best_ratio[b, k, local_j, local_i] != 0:
                            if box_best_ratio[b, k, local_j, local_i] > ratio[mask]:
                                y_true[b, k, local_j, local_i, :] = 0
                            else:
                                continue

                        #----------------------------------------#
                        # Extract ground truth box type
                        #----------------------------------------#
                        c = batch_target[t, 4].long()

                        #----------------------------------------#
                        # noobj_mask represents feature points without a target
                        #----------------------------------------#
                        noobj_mask[b, k, local_j, local_i] = 0
                        #----------------------------------------#
                        # tx, ty represent the true values of the center adjustment parameters.
                        #----------------------------------------#
                        y_true[b, k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[b, k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[b, k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[b, k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[b, k, local_j, local_i, 4] = 1
                        y_true[b, k, local_j, local_i, c + 5] = 1
                        #----------------------------------------#
                        # Get the best ratio for the current prior bounding box
                        #----------------------------------------#
                        box_best_ratio[b, k, local_j, local_i] = ratio[mask]

        return y_true, noobj_mask

    def get_pred_boxes(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w):
        #-----------------------------------------------------#
        # Calculate the total number of images
        #-----------------------------------------------------#
        bs = len(targets)

        #-----------------------------------------------------#
        # Generate grid, prior box center, top-left corner of the grid
        #-----------------------------------------------------#
        grid_x = torch.linspace(0, in_w - 1,
                                in_w).repeat(in_h, 1).repeat(int(bs * len(self.anchors_mask[l])), 1,
                                                             1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1,
                                in_h).repeat(in_w,
                                             1).t().repeat(int(bs * len(self.anchors_mask[l])), 1,
                                                           1).view(y.shape).type_as(x)

        # Generate prior boxes' width and height
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        #-------------------------------------------------------#
        # Calculate adjusted prior box center and dimensions.
        #-------------------------------------------------------#
        pred_boxes_x = torch.unsqueeze(x * 2. - 0.5 + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y * 2. - 0.5 + grid_y, -1)
        pred_boxes_w = torch.unsqueeze((w * 2)**2 * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze((h * 2)**2 * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)
        return pred_boxes


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau)
                                        )  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


def weights_init(net, init_type='normal', init_gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' %
                                          init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type,
                     lr,
                     min_lr,
                     total_iters,
                     warmup_iters_ratio=0.05,
                     warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05,
                     step_num=10):

    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter,
                          iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters),
                                              2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi * (iters - warmup_total_iters) /
                               (total_iters - warmup_total_iters - no_aug_iter)))
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate**n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters,
                       warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr)**(1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
