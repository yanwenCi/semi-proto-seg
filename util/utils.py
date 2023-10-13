"""Util functions"""
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
from torch import nn
import numpy as np


def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

CLASS_LABELS = {
    'VOC': {
        'all': set(range(1, 21)),
        0: set(range(1, 21)) - set(range(1, 6)),
        1: set(range(1, 21)) - set(range(6, 11)),
        2: set(range(1, 21)) - set(range(11, 16)),
        3: set(range(1, 21)) - set(range(16, 21)),
    },
    'COCO': {
        'all': set(range(1, 81)),
        0: set(range(1, 81)) - set(range(1, 21)),
        1: set(range(1, 81)) - set(range(21, 41)),
        2: set(range(1, 81)) - set(range(41, 61)),
        3: set(range(1, 81)) - set(range(61, 81)),
    }
}

def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 1

    for i in cls_ids:
        mask_idx = np.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 0
    return fg_bbox, bg_bbox



class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x, y=None):
        x= [transforms.ToPILImage()(x_) for x_ in x.cpu().data]
        #angle = random.choice(self.angles)
        x=[TF.rotate(x_, self.angles) for x_ in x]
        x = [transforms.ToTensor()(x_)[np.newaxis, ...] for x_ in x]
        if y is not None:
            y = [transforms.ToPILImage()(x_) for x_ in y.cpu().data]
            y=[TF.rotate(y_, self.angles) for y_ in y]
            y=[transforms.ToTensor()(y_)[np.newaxis,...] for y_ in y]
            return torch.cat(x,dim=0).cuda(),torch.cat(y,dim=0).cuda()
        else:
            return torch.cat(x,dim=0).cuda()

def transforms():
    data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    ])
    return data_transforms

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(input_s, input_q,  beta=0.5, cutmix_prob=0.5):
    r = np.random.rand(1)
    if beta > 0 and r < cutmix_prob:
        # generate mixed sample
        lam = np.random.beta(beta, beta)  # 随机的lam
        rand_index = torch.randperm(input_q.size()[0])  # 在batch中随机抽取一个样本，记为i
        bbx1, bby1, bbx2, bby2 = rand_bbox(input_q.size(), lam)  # 随机产生一个box的四个坐标
        input_q[:, :, bbx1:bbx2, bby1:bby2] = input_s[rand_index, :, bbx1:bbx2, bby1:bby2]
    return input_q


def evaluate (label_pred, label_gt, n_class):

    epsilon = 1.0e-6
    assert len(label_gt) == len(label_pred)
    batchSize = len(label_gt)
    dice_scores = np.zeros((batchSize, n_class), dtype=np.float32)
    for batch_id, (l_gt, l_pred) in enumerate(zip(label_gt, label_pred)):
        for class_id in range(n_class):
            img_A = np.array(l_gt == class_id, dtype=np.float32).flatten()
            img_B = np.array(l_pred == class_id, dtype=np.float32).flatten()
            score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
            dice_scores[batch_id, class_id] = score

    return np.mean(dice_scores, axis=0)



def sigmoid_rampup(current_epoch):
        current = np.clip(current_epoch, 0.0, 5.0)
        phase = 1.0-current / 5.0
        return np.exp(-5.0 * phase * phase).astype(np.float32)

def linear_rampup(current, rampup_length, clip=1.0):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0*clip
    else:
        return clip*current / rampup_length

class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes ,logit=False, a=0.5):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.alpha=a
        self.logit = logit

    def forward(self,input, target):
        # input prediction B*C*H*W
        # target label B*1*H*W
        smooth = 0.01
        batch_size = input.size(0)
        if self.logit:
            if self.n_classes==1:
                input = torch.sigmoid(input)
            else:
                input = torch.softmax(input, dim=1)
                target = self.one_hot_encoder(target).contiguous()#.view(batch_size, self.n_classes, -1)
        target = target.view(batch_size, self.n_classes, -1)
        input=input.view(batch_size, self.n_classes, -1)
        #w= torch.Tensor([1-self.alpha, self.alpha]).unsqueeze(dim=0).repeat(batch_size, 1).cuda()
        inter =  torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth
        scores=2.0 * (inter / union)

        score = 1.0 - torch.sum(scores) / (self.n_classes*float(batch_size))
        #* float(self.n_classes))
        #print(scores, score)
        return score

class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)

    def forward(self, X_in):
        n_dim = X_in.dim()
        self.ones = self.ones.to(X_in.device)
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().contiguous().view(num_element)
        out = self.ones.index_select(0, X_in).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

