"""
Fewshot Semantic Segmentation
"""
import copy
from collections import OrderedDict
from info_nce import InfoNCE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .networks import conv_block, up_conv
#from .networks import Encoder
#from .networks import U_Net
#from .networks import resnet18

class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model
    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, in_channels=3, out_nc=2, cfg=None, pretrained_path=None, cutmix=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.cutmix = cutmix 
        # Encoder
        self.encoder = Encoder(in_ch=in_channels, n1=64)
        self.decoder = DecoderU(n1=64, out_nc=out_nc)
        self.encoder_ema = copy.deepcopy(self.encoder)
        self.decoder_ema = copy.deepcopy(self.decoder)
        self.set_requires_grad(self.decoder_ema, requires_grad=False)
        self.set_requires_grad(self.encoder_ema, requires_grad=False)
        self.dice_loss = SoftDiceLoss(n_classes=2, logit=True)
        #self.encoder = nn.Sequential(OrderedDict([('backbone', Encoder(in_channels, self.pretrained_path)),]))
        self.loss_nce = InfoNCE(negative_mode='unpaired')

    def update_ema_variables(self, model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, step=None, istrain=True):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs], dim=0)
        imgs_qry=     torch.cat(qry_imgs, dim=0)


        se1,se2,se3,se4,supp_fts_inter= self.encoder(imgs_concat)
        fts_size = supp_fts_inter.shape[-2:]
        #supp_fts_inter = img_fts[:n_ways * n_shots * batch_size]
        supp_fts = supp_fts_inter.view(n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        if self.cutmix:
            imgs_qry = self.cutmix(imgs_concat, imgs_qry)
            qe1,qe2,qe3,qe4,qry_fts_inter = self.encoder(imgs_qry)
        else:
            qe1,qe2,qe3,qe4,qry_fts_inter = self.encoder(imgs_qry)
        #qry_fts_inter = img_fts[n_ways * n_shots * batch_size:]
        qry_fts = qry_fts_inter.view(n_queries, batch_size, -1, *fts_size)  # N x B x C x H' x W'


        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H x W
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H x W


        ###### Compute loss ######
        align_loss, nce_loss = 0,0
        outputs, supp_map = [],[]
        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]

            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

            ###### Compute the distance ######
            prototypes = [bg_prototype,] + fg_prototypes
            dist = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True))

            ###### Prototype alignment loss ######
            if self.config['align']: #and self.training:
                align_loss_epi, supp_epi, qry_prototypes = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi
                supp_map.append(supp_epi)
                bg_prototypes=torch.cat((bg_prototype, qry_prototypes[[0]]),dim=0)
                nce_loss += self.loss_nce(qry_prototypes[[1]], fg_prototypes[0], bg_prototypes)
        supp_map=torch.stack(supp_map, dim=1).view(-1, *supp_epi.shape[1:])   
        
        qry_map = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        qry_map = qry_map.view(-1, *qry_map.shape[2:])
        #dnsp=nn.Sequential(nn.Softmax(dim=1), nn.AdaptiveAvgPool2d(qry_fts.shape[3:]))
        supp_fts_f, supp_fts_b = self.unpooling(supp_fts_inter, qry_prototypes[[1]], qry_prototypes[[0]])
        supp_seg = self.decoder(se1,se2,se3,se4,supp_fts_inter, supp_fts_f, supp_fts_b)
        #supp_seg = self.decoder(torch.cat((supp_fts_inter,  dnsp(supp_map)), dim=1))
        #supp_seg = self.decoder(torch.cat((supp_fts_inter,  qry_fts_inter, dnsp(supp_map)), dim=1))
        if step is not None:
            if istrain:
                noise = torch.empty(torch.cat(qry_imgs, dim=0).shape).normal_(mean=0, std=0.1).cuda()
                self.update_ema_variables(self.encoder, self.encoder_ema, 0.999, step)
                self.update_ema_variables(self.decoder, self.decoder_ema, 0.999, step)
            else:
                noise = torch.zeros_like(torch.cat(qry_imgs, dim=0)).cuda()
            qe1e, qe2e, qe3e, qe4e, qry_fts_ema = self.encoder_ema(torch.cat(qry_imgs, dim=0)+noise)  # ema query features
            
            #qry_seg_ema = self.decoder_ema(qry_fts_ema)
            qry_fts_ema_f, qry_fts_ema_b = self.unpooling(qry_fts_ema,  fg_prototypes[0], bg_prototype)
            qry_seg_ema = self.decoder_ema(qe1e, qe2e, qe3e, qe4e, qry_fts_ema, qry_fts_ema_f, qry_fts_ema_b)#(torch.cat((qry_fts_ema,  qry_fts_inter, dnsp(qry_map)), dim=1))
        else:
            qry_seg_ema=None,None

        if self.cutmix:
            qry_seg_ema = self.cutmix(supp_seg, qry_seg_ema)
        return qry_map, align_loss / batch_size, supp_seg, qry_seg_ema, nce_loss, prototypes


    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        return dist


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
        return masked_fts
    
    def unpooling(self, fts, proto_f, proto_b):
        proto_f = F.interpolate(proto_f[:,:,None, None], size=fts.shape[-2:], mode='nearest')
        proto_b = F.interpolate(proto_b[:,:,None, None], size=fts.shape[-2:], mode='nearest')
        return fts+proto_f, fts-proto_f
        

    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype
        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype


    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch
        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding features for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        #skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # (1 + Wa) x C

        # Compute the support loss
        loss = 0
        outputs=[]
        for way in range(n_ways):
            #if way in skip_ways:
               # continue
            # Get the query prototypes
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:],
                                          mode='bilinear', align_corners=True)
                outputs.append(supp_pred)
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                #loss = loss + F.cross_entropy(supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
                loss += self.dice_loss(supp_pred, supp_label[None, ...]) / n_shots / n_ways
        supp_map = torch.stack(outputs, dim=1).view(-1,*supp_pred.shape[1:])  # N x (1 + Wa) x H x W
        #supp_map = supp_map.view(-1, *supp_map.shape[1:])
        return loss, supp_map, qry_prototypes


def evaluate (label_gt, label_pred, n_class):

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
            input = F.softmax(input, dim=1)
        input=input.view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        w= torch.Tensor([1-self.alpha, self.alpha]).unsqueeze(dim=0).repeat(batch_size, 1).cuda()
        inter = w*torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth
        scores=2.0 * (inter / union)

        score = 1.0 - torch.sum(scores) / (float(batch_size) )
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


class Decoder(nn.Module):
    def __init__(self,  ngf,n_downsampling=3, out_nc=2,  sup=False):
        super().__init__()

        self.sup = sup
        # freeze the backbone
        mult = 2 ** (n_downsampling)
        model_dec = []
        self.fuse1 = nn.Sequential(nn.Conv2d(mult*ngf,1, kernel_size=1, stride=1, bias=False), nn.Sigmoid())
        self.fuse2 = nn.Sequential(nn.Conv2d(mult*ngf,1, kernel_size=1, stride=1, bias=False), nn.Sigmoid())
        model_dec = [nn.Conv2d(2+mult*ngf, mult*ngf, kernel_size=1, stride=1, bias=False), nn.ReLU()]
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model_dec += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=True),
                          nn.BatchNorm2d(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model_dec += [nn.ReflectionPad2d(3)]
        model_dec += [nn.Conv2d(ngf, ngf, kernel_size=7, padding=0)]
        model_dec += [nn.ReLU(True)]
        model_dec += [nn.Conv2d(ngf, out_nc, kernel_size=1, padding=0)]#, nn.Softmax(dim=1)]
        self.model_dec1 = nn.Sequential(*model_dec)


    def forward(self, inputs, proto_f, proto_b):
        # noise = torch.empty_like(inputs).normal_(mean=0, std=0.1).cuda()
        fuse_f = self.fuse1(proto_f)
        fuse_b = self.fuse2(proto_b)
        out_q = self.model_dec1(torch.cat((inputs, fuse_f, fuse_b), dim=1))
        return out_q


class Encoder(nn.Module):
    def __init__(self, in_ch, n1):
        super().__init__()
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        return e1, e2, e3, e4, e5


class DecoderU(nn.Module):
    def __init__(self,n1,  out_nc=2):
        super().__init__()
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_nc, kernel_size=1, stride=1, padding=0)
        self.fuse1 = nn.Sequential(nn.Conv2d(filters[4],1, kernel_size=1, stride=1, bias=False), nn.Sigmoid())
        self.fuse2 = nn.Sequential(nn.Conv2d(filters[4],1, kernel_size=1, stride=1, bias=False), nn.Sigmoid())
        self.model_dec = nn.Sequential(nn.Conv2d(filters[4]+2, filters[4], kernel_size=1, stride=1, bias=False), nn.ReLU())
    def forward(self, e1,e2,e3,e4,e5, proto_f,proto_b):
        fuse_f = self.fuse1(proto_f)
        fuse_b = self.fuse2(proto_b)
        e5 = self.model_dec(torch.cat((e5, fuse_f, fuse_b), dim=1))
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out
