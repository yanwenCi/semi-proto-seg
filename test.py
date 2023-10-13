"""Evaluation Script"""
import os
import shutil
from itertools import cycle
import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
#from models.fewshot_relation import FewShotSeg
#from models.fewshot_unetcat import FewShotSeg
from models.fewshot_deeplab import FewShotSeg
#from models.fewshot_inter3 import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import ToTensorNormalize
from dataloaders.transforms import Resize, DilateScribble
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox
from config import ex
from dataloaders import *
from util.metric import positive_lesion_rate
from utils import cutmix

@ex.automain
def main(_run, _config, _log):
    for source_file, _ in _run.experiment_info['sources']:
        os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                    exist_ok=True)
        _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    os.makedirs(f'{_run.observers[0].dir}/savefigs', exist_ok=True)

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    #torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)
    n_shot=_config['task']['n_shots']
    n_way = _config['task']['n_ways']
    batch_size=_config['batch_size']
    n_query = _config['task']['n_queries']
    n_label = _config['n_label']
    _log.info('###### Create model ######')
    in_nc, out_nc = _config['in_nc'], _config['out_nc']
    model = FewShotSeg(in_channels=in_nc, out_nc=out_nc,pretrained_path=_config['path']['init_path'], cfg=_config['model'])#, cutmix=None)
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    #if not _config['notrain']:
    model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()


    _log.info('###### Prepare data ######')
    #dataloader_test = CustomDatasetDataLoader(dataroot=_config['dataroot_test'], split='test', batch_size=batch_size,
    #                                          transform=None, label_num=None).load_data()
    #dataloader_support = CustomDatasetDataLoader(dataroot=_config['dataroot_support'], split='test', batch_size=batch_size*n_shot*n_way,
    #                                          transform=None, label_num=n_label).load_data()
    dataloader_test = CustomDatasetDataLoader(dataroot=_config['dataroot_test'], split='test_query_shot1', batch_size=batch_size,
                                              transform=None, label_num=None).load_data()
    dataloader_support = CustomDatasetDataLoader(dataroot=_config['dataroot_support'], split='test_support_shot1', batch_size=batch_size*n_shot*n_way,
                                              transform=None, label_num=n_label).load_data()
    _log.info('###### Testing begins ######')
    max_label=1
    #proto = np.load(_config['snapshot'].replace('best.pth', 'proto.npz'))
    pred_positive_lesions_sum, pred_lesions_sum, gt_lesions_sum, gt_positive_lesions_sum=0,0,0,0
    labels=list(range(max_label+1))
    metric = Metric(max_label=max_label, n_runs=_config['n_runs'])
    with torch.no_grad():
        for run in range(_config['n_runs']):
            _log.info(f'### Run {run + 1} ###')
            set_seed(_config['seed'] + run)

            for i, (sample_batched, sample_query) in tqdm.tqdm(enumerate(zip(cycle(dataloader_support), dataloader_test))):
                img_size=sample_batched['image'].shape[2:]
                support_images =  sample_batched['image'].view(n_way, n_shot, batch_size, in_nc,  *img_size).cuda()
                support_images =  [[img__ for img__ in img] for img in support_images] #ways*shots*[B*3*H*W], ways=1 here
                support_mask = [way.view((n_way, n_shot, batch_size) + img_size ).cuda() for idx, way in sample_batched['label'].items()]
                support_mask = [[[img__ for img__ in  img] ]for mask in support_mask for img in mask]
                support_mask_t=sample_batched['inst'].type(torch.LongTensor).cuda()
                query_images = sample_query['image'].view(n_query, batch_size, in_nc, *img_size ).cuda()
                query_images = [img for img in query_images] #shots*[B*3*H*W],
                query_labels = sample_query['inst'].type(torch.LongTensor).cuda()#.view((n_query, batch_size) + img_size)
                query_names = sample_query['prefix']
                query_pred, _, _,query_pred2,_,_ = model(support_images, support_mask[1], support_mask[0], query_images, step=0, istrain=False)
                #query_pred2 = query_pred.argmax(dim=1)
                if query_pred2.shape[1]==1:
                    query_pred2 = torch.sigmoid(query_pred2.squeeze()).round()
                else:
                    query_pred2 = query_pred2.argmax(dim=1)
                for jj in range(query_pred.shape[0]):
                    #print(query_pred2[jj,1,...].shape, query_labels[jj,0,...].shape)
                    _,pred_positive_lesion, pred_lesion = positive_lesion_rate(query_pred2[jj,...].cpu().data.numpy(), query_labels[jj,...].cpu().data.numpy(), [0.1])
                    pred_positive_lesions_sum += pred_positive_lesion[0]
                    pred_lesions_sum += pred_lesion
                    _,gt_positive_lesion, gt_lesion = positive_lesion_rate(query_labels[jj,...].cpu().data.numpy(), query_pred2[jj,...].cpu().data.numpy(), [0.1])
                    gt_positive_lesions_sum += gt_positive_lesion[0]
                    gt_lesions_sum += gt_lesion
                    #print(pred_positive_lesion, pred_lesion)
#                if i%100==0:
#                    import matplotlib.pyplot as plt
#                    plt.subplot(2, 2, 2)
#                    plt.imshow(query_images[0][0,0, ...].cpu().data, cmap='gray')
#                    plt.subplot(2,2,2)
#                    plt.imshow(query_pred.argmax(dim=1)[0,...].cpu().data, cmap='gray')
#                    plt.subplot(2, 2, 3)
#                    plt.imshow(query_pred2.argmax(dim=1)[0, ...].cpu().data, cmap='gray')
#                    plt.subplot(2, 2, 4)
#                    plt.imshow(query_labels[0, ...].cpu().data, cmap='gray')
#                    plt.savefig(f'{_run.observers[0].dir}/savefigs/{query_names[0]}')
                    metric.record(np.array(query_pred2[jj].cpu()),
                              np.array(query_labels[jj].cpu()),
                              labels=None, n_run=run)

            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classdice, inter_stddice = metric.get_dice(labels=sorted(labels), n_run=run)
            #classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)
            lesion_precision=pred_positive_lesions_sum/pred_lesions_sum
            lesion_recall = gt_positive_lesions_sum/gt_lesions_sum
            _run.log_scalar('classIoU', classIoU.tolist())
            _run.log_scalar('meanIoU', meanIoU.tolist())
            _run.log_scalar('lesionPrecision', lesion_precision)
            _run.log_scalar('lesionRecall', lesion_recall)
            #_run.log_scalar('classIoU_binary', classIoU_binary.tolist())
            #_run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())
            _run.log_scalar('classdice', classdice.tolist())
            _run.log_scalar('stddice', inter_stddice.tolist())
            _log.info(f'classIoU: {classIoU}')
            _log.info(f'meanIoU: {meanIoU}')
            _log.info(f'lesionPrecision: {lesion_precision}')
            _log.info(f'lesionRecall: {lesion_recall}')
            #_log.info(f'classIoU_binary: {classIoU_binary}')
            #_log.info(f'meanIoU_binary: {meanIoU_binary}')
            _log.info(f'classdice: {classdice}')
            _log.info(f'stddice: {inter_stddice}')

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
    classdice, classdice_std, meandice, meandice_std = metric.get_dice_batch(labels=sorted(labels))
    #classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()
    
    _log.info('----- Final Result -----')
    _run.log_scalar('final_classIoU', classIoU.tolist())
    _run.log_scalar('final_classIoU_std', classIoU_std.tolist())
    _run.log_scalar('final_meanIoU', meanIoU.tolist())
    _run.log_scalar('final_meanIoU_std', meanIoU_std.tolist())
    #_run.log_scalar('final_classIoU_binary', classIoU_binary.tolist())
    #_run.log_scalar('final_classIoU_std_binary', classIoU_std_binary.tolist())
    #_run.log_scalar('final_meanIoU_binary', meanIoU_binary.tolist())
    #_run.log_scalar('final_meanIoU_std_binary', meanIoU_std_binary.tolist())
    _run.log_scalar('final_classdice', classdice.tolist())
    _run.log_scalar('final_classdice_std', classdice_std.tolist())
    _run.log_scalar('final_meandice', meandice.tolist())
    _run.log_scalar('final_meandice_std', meandice_std.tolist())
    _log.info(f'classIoU mean: {classIoU}')
    _log.info(f'classIoU std: {classIoU_std}')
    _log.info(f'meanIoU mean: {meanIoU}')
    _log.info(f'meanIoU std: {meanIoU_std}')
    _log.info(f'classdice mean: {classdice}')
    _log.info(f'classdice std: {classdice_std}')
    _log.info(f'meandice mean: {meandice}')
    _log.info(f'meandice std: {meandice_std}')
    #_log.info(f'classIoU_binary mean: {classIoU_binary}')
    #_log.info(f'classIoU_binary std: {classIoU_std_binary}')
    # _log.info(f'meanIoU_binary mean: {meanIoU_binary}')
    #_log.info(f'meanIoU_binary std: {meanIoU_std_binary}')
