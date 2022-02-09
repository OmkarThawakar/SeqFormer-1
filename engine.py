# ------------------------------------------------------------------------
# Train and eval functions used in main.py
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import os
import sys
from typing import Iterable
import cv2
import numpy as np
import json
import copy

import torch
import util.misc as utils
from util.misc import NestedTensor
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from PIL import Image, ImageDraw

from scipy.optimize import linear_sum_assignment

from models import build_model
import torchvision.transforms as T
import torch.nn.functional as F

import pycocotools.mask as mask_util
from pathlib import Path

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 4000

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    i = 0

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):

        i+=1

        if i<2 :
            outputs, loss_dict = model(samples, targets, criterion, train=True)

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
        
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            
            optimizer.step()
    
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(grad_norm=grad_total_norm)
            samples, targets = prefetcher.next()

        else:
            break
     
    torch.cuda.empty_cache()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def ytvis_evaluate(model_path, device, args, epoch_num):

    transform = T.Compose([
                T.Resize(360),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


    args.masks=True
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    state_dict = torch.load(model_path)['model']
    model.load_state_dict(state_dict)
    model.eval()
    folder = args.ytvis_eval_img_path
    videos = json.load(open(args.ytvis_eval_ann_path,'rb'))['videos']#[:5]
    # videos = [videos[1],videos[8],videos[22],videos[34]]
    vis_num = len(videos)
    # postprocess = PostProcessSegm_ifc()
    result = [] 
    for i in range(1):
        
        id_ = videos[i]['id']
        #if i == 2 :
        print("Process video: ",i)
        vid_len = videos[i]['length']
        file_names = videos[i]['file_names']
        video_name_len = 10 

        pred_masks = None
        pred_logits = None

        img_set=[]
        for k in range(vid_len):
            im = Image.open(os.path.join(folder,file_names[k]))
            w, h = im.size
            sizes = torch.as_tensor([int(h), int(w)])
            img_set.append(transform(im).unsqueeze(0).cuda())

        img = torch.cat(img_set,0)

        n, c, H, W = img.shape
    
        model.detr.num_frames=vid_len  

        outputs, memory, max_feature = model.inference(img,img.shape[-1],img.shape[-2])

        logits = outputs['pred_logits'][0]
        output_mask = outputs['pred_masks'][0]
        output_boxes = outputs['pred_boxes'][0]

        
        H = output_mask.shape[-2]
        W = output_mask.shape[-1]


        scores = logits.sigmoid().cpu().detach().numpy()
        hit_dict={}

        topkv, indices10 = torch.topk(logits.sigmoid().cpu().detach().flatten(0),k=10)
        indices10 = indices10.tolist()
        for idx in indices10:
            queryid = idx//42
            if queryid in hit_dict.keys():
                hit_dict[queryid].append(idx%42)
            else:
                hit_dict[queryid]= [idx%42]


        for inst_id in hit_dict.keys():
            masks = output_mask[inst_id]
            pred_masks = F.interpolate(masks[:,None,:,:], (im.size[1],im.size[0]),mode="bilinear")
            pred_masks = pred_masks.sigmoid().cpu().detach().numpy()>0.5  #shape [100, 36, 720, 1280]
            # if pred_masks.max()==0:
            #     print('skip')
            #     continue
            for class_id in hit_dict[inst_id]:
                category_id = class_id
                score =  scores[inst_id,class_id]
        #       print('name:',CLASSES[category_id-1],', score',score)
                instance = {'video_id':id_, 'video_name': file_names[0][:video_name_len], 'score': float(score), 'category_id': int(category_id)}  
                segmentation = []
                for n in range(vid_len):
                    if score < 0.001:
                        segmentation.append(None)
                    else:
                        mask = (pred_masks[n,0]).astype(np.uint8) 
                        rle = mask_util.encode(np.array(mask[:,:,np.newaxis], order='F'))[0]
                        rle["counts"] = rle["counts"].decode("utf-8")
                        segmentation.append(rle)
                instance['segmentations'] = segmentation
                result.append(instance)

    
    output_dir = Path(args.output_dir)
                    
    with open(output_dir / f'results_epoch_{epoch_num:04}.json', 'w', encoding='utf-8') as f:
        json.dump(result,f)

    ##### Evaluate the Results ######

    from pycocotools.ytvos import YTVOS
    from ytvos_eval import YTVOSeval


    annType = ['segm','bbox']
    annType = annType[0]      #specify type here
    print('Running demo for *%s* results.'%(annType))

    annFile = args.ytvis_eval_ann_path
    ytvosGt=YTVOS(annFile)

    resFile = output_dir / f'results_epoch_{epoch_num:04}.json'
    ytvosDt=ytvosGt.loadRes(str(resFile))

    vidIds=sorted(ytvosGt.getVidIds())

    # running evaluation
    ytvosEval = YTVOSeval(ytvosGt,ytvosDt,annType)
    print('='*50)
    ytvosEval.params.vidIds  = vidIds
    print('-'*50)
    ytvosEval.evaluate()

    print('='*50)
    print('Eval Stats : ', ytvosEval.stats )
    print('='*50)

    return ytvosEval.stats, ytvosEval
