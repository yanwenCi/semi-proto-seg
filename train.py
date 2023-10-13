"""Training Script"""
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
import albumentations as A
from models.fewshot_unet import FewShotSeg
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from config import ex
from dataloaders import *
from itertools import cycle
import torch.multiprocessing
from util.utils import cutmix, SoftDiceLoss, evaluate, linear_rampup
torch.multiprocessing.set_sharing_strategy('file_system')

@ex.automain
def main( _run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)
    best_acc=0

    _log.info('###### Create model ######')
    #model = FewShotSeg( cfg=_config['model'])
    out_nc=_config['out_nc']
    in_nc = _config['in_nc']
    model = FewShotSeg(in_channels=in_nc, out_nc=out_nc, pretrained_path=_config['path']['init_path'], cfg=_config['model'], cutmix=cutmix)

    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    #model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.train()


    _log.info('###### Load data ######')
    transform = A.Compose([
        A.OneOf([A.Flip(p=0.5),
                 A.RandomRotate90(p=0.5), ], p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.25),
        A.OneOf([
            A.Blur(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.8), ], p=0.25),
    ],
        additional_targets={
            'image': 'image',
            'mask': 'mask',
            'mask1': 'mask'}
    )
    #transform=None
    n_shot=_config['task']['n_shots']
    n_way = _config['task']['n_ways']
    n_label=_config['n_label']
    batch_size=_config['batch_size']
    n_query = _config['task']['n_queries']
    dataloader_support = CustomDatasetDataLoader(dataroot=_config['dataroot_support'], split='valid', batch_size=batch_size*n_shot,
                                              transform=transform, catId=2, label_num=n_label).load_data()
    dataloader_query =  CustomDatasetDataLoader(dataroot=_config['dataroot_query'], split='train', batch_size=batch_size*n_query,
                                              transform=transform, catId=2, label_num=None).load_data()
    dataloader_test = CustomDatasetDataLoader(dataroot=_config['dataroot_test'], split='test', batch_size=batch_size*n_query,
                                              transform=None, catId=2, label_num=None).load_data()

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion_CE = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])
    criterion = SoftDiceLoss(n_classes=out_nc, logit=True)
    __Softmax=torch.nn.Softmax(dim=1)
    criterion_consist = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])#= SoftDiceLoss(n_classes=2, logit=True)#nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])#nn.MSELoss()
    i_iter = 0

    _log.info('###### Training ######')
    step=0
    loss_param=1
    patience=0
    len_data = max(len(dataloader_support), len(dataloader_query))
    for epo in range(_config['epoch']):
        log_loss = {'query_loss': 0, 'align_loss': 0, 'support_loss':0, 'nce_loss':0}
        #for i in range(1000):#max(len(dataloader_support), len(dataloader_query))):
        for i_iter, (sample_batched, sample_query,_) in enumerate(zip(cycle(dataloader_support), cycle(dataloader_query), range(len_data))):
        #for i_iter, (sample_batched, sample_query) in enumerate(zip(cycle(dataloader_support), dataloader_query)):
    #        sample_batched=enumerate(dataloader_support)
    #        sample_query = enumerate(dataloader_query)
            # Prepare input
            step+=1#i_iter+epo*len(dataloader_query)
            img_size=sample_batched['image'].shape[2:]
            support_images =  sample_batched['image'].view(1, n_shot, batch_size, in_nc,  *img_size).cuda()
            support_images =  [[img__ for img__ in img] for img in support_images] #ways*shots*[B*3*H*W], ways=1 here
            support_mask = [way.view(n_way, n_shot, batch_size, *img_size ).cuda()
                           for idx, way in sample_batched['label'].items()]
            support_mask = [[[img__ for img__ in img]  for img in mask] for mask in support_mask]
            support_mask_t= sample_batched['inst'].type(torch.LongTensor).cuda()
            #support_mask_t = [[[img__ for img__ in  img] ]for img in support_mask_t]
            #support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
            #                   for way in sample_batched['support_mask']]

            query_images = sample_query['image'].view(n_query, batch_size, in_nc, *img_size ).cuda()
            query_images = [img for img in query_images] #shots*[B*3*H*W],
            query_labels = sample_query['inst'].type(torch.LongTensor).cuda()#.view((n_query, batch_size) + img_size)
            #query_labels = [[img] for img in query_labels]

            # Forward and Backward
            optimizer.zero_grad()
            qry_proto_seg, align_loss, supp_mt_seg, qry_mt_seg, nce_loss, proto = model(support_images, support_mask[1], support_mask[0],
                                       query_images, step)

            support_loss = criterion(supp_mt_seg, support_mask_t)#+criterion_CE(supp_mt_seg, support_mask_t)
            #query_loss = criterion_consist(qry_proto_seg,query_labels)
            if out_nc==1:
                pseudo_query_mask = torch.round(torch.sigmoid(qry_mt_seg.squeeze())).type(torch.LongTensor).cuda()
            else:
                pseudo_query_mask = qry_mt_seg.argmax(dim=1)
            query_loss = criterion_consist(qry_proto_seg, pseudo_query_mask)#.argmax(dim=1))# + criterion_CE(qry_proto_seg, __Softmax(qry_mt_seg).argmax(dim=1))
            #print(supp_seg.max(),supp_seg.min(), support_mask_t.max())
            #loss =  support_loss + query_loss
            #if epo>0:
            #    loss_param=0.2
            loss_param = linear_rampup(step, 5000, 1)# sigmoid_rampup(epo) 
            #loss = align_loss * _config['align_loss_scaler']+ nce_loss * 0.1 #+ query_loss + support_loss
            loss = loss_param*query_loss + support_loss +  align_loss * _config['align_loss_scaler']+ nce_loss * 0.1
            loss.backward()
            optimizer.step()
            scheduler.step()
            curr_lr = scheduler.get_last_lr()[0]
            #print(curr_lr)
            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            support_loss = support_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
            _run.log_scalar('query_loss', query_loss)
            _run.log_scalar('support_loss', support_loss)
            _run.log_scalar('align_loss', align_loss)
            _run.log_scalar('nce_loss', query_loss)
            log_loss['query_loss'] += query_loss
            log_loss['support_loss'] += support_loss
            log_loss['align_loss'] += align_loss
            log_loss['nce_loss'] += nce_loss


            # print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:
                query_loss = log_loss['query_loss'] / (i_iter + 1)
                align_loss = log_loss['align_loss'] / (i_iter + 1)
                support_loss = log_loss['support_loss'] / (i_iter + 1)
                nce_loss = log_loss['nce_loss'] / (i_iter + 1)
                print(f'epoch {epo}, step {i_iter+1}, lr {curr_lr:.4f}, lossparam {loss_param:.2f}: query_loss: {query_loss:.4f}, '
                      f'align_loss: {align_loss:.4f}, support_loss:{support_loss:.4f}, nce_loss:{nce_loss:.4f}')

            if (i_iter + 1) % _config['save_pred_every'] == 0:
                _log.info('###### Taking snapshot ######')
                #torch.save(model.state_dict(),
                #       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
                test_dice, test_dice2 =0,0

                for ii, sample_test in enumerate(dataloader_test):
                    if ii >500:
                        break
                    test_images = sample_test['image'].view(n_query, batch_size, in_nc, *img_size).cuda()
                    test_name = sample_test['prefix']
                    test_images = [img for img in test_images]  # shots*[B*3*H*W],
                    test_labels = sample_test['inst'].type(
                        torch.LongTensor) # .view((n_query, batch_size) + img_size)
                    with torch.no_grad():
                        test_pred, _, _,test_pred2,_i, _ = model(support_images, support_mask[1], support_mask[0],
                                       test_images, step, istrain=False)
                    test_dice += evaluate(__Softmax(test_pred).argmax(dim=1).cpu().data, test_labels, n_class=2)[1]
                    if out_nc>1:
                        test_dice2 += evaluate(__Softmax(test_pred2).argmax(dim=1).cpu().data, test_labels, n_class=2)[1]
                    else:
                        test_dice2 += evaluate(torch.round(torch.sigmoid(test_pred2.squeeze())).cpu().data, test_labels, n_class=2)[1]    
                test_dice/=ii
                test_dice2/=ii#(len(dataloader_test)
                patience+=1
                if test_dice2>best_acc:
                    patience=0
                    best_acc=test_dice2
                    #np.savez(os.path.join(f'{_run.observers[0].dir}/snapshots', f'proto.npz'), proto)
                    #torch.save(model.state_dict(),
                    #   os.path.join(f'{_run.observers[0].dir}/snapshots', f'best_{best_acc:.4f}.pth'))
                    torch.save(model.state_dict(), os.path.join(f'{_run.observers[0].dir}/snapshots', f'best.pth')) 
                print(f'step {i_iter + 1}: acc: {test_dice:.4f}, acc2: {test_dice2:.4f}')
        if patience>_config['save_pred_every']*10:
            print('early stopping at epoch {epoch}')
            break
    _log.info('###### Saving final model ######')
        #torch.save(model.state_dict(),
        #       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
    torch.save(model.state_dict(),   os.path.join(f'{_run.observers[0].dir}/snapshots', f'best_{best_acc:.4f}.pth'))



