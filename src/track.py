from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy.core._multiarray_umath import ndarray

import _init_paths
import os
import os.path as osp
import shutil
import cv2
import json
import logging
import argparse
import motmetrics as mm
from tqdm import tqdm
import numpy as np
import torch

from collections import defaultdict
from lib.tracker.multitracker import JDETracker, MCJDETracker
from lib.tracker.YoloTracker import YOLOTracker
from lib.tracker.YoloByteTracker import YOLOBYTETracker

from lib.tracking_utils import visualization as vis
from lib.tracking_utils.log import logger
from lib.tracking_utils.timer import Timer
from lib.tracking_utils.evaluation import Evaluator
from lib.tracking_utils.utils import mkdir_if_missing

import lib.datasets.yolomot as datasets

from lib.opts import opts


class_names = ["pedestrian"]


def write_results(filename, results, data_type, img_dim=(375, 1242), bbox_dim=(576, 1024), padding=(0,0)):
    
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    else:
        raise ValueError(data_type)
    
    im_h, im_w = img_dim
    bbox_h, bbox_w = bbox_dim
    
    with open(filename, 'w') as file:
        # Each Class Has List of Frame Index
        for class_id, frames in results.items():
            
            # Convert BDD results to only KITTI classes
            if class_id == 0:
                cls_name = 'pedestrian'
            elif class_id == 2:
                cls_name = 'car'
            else:
                continue
            
            # Each Frame is (frameIndex, list(bboxes), list(track IDs), list(scores))
            for (frame, bboxes, track_ids, scores) in frames:
                for tlwh, track_id, score in zip(bboxes, track_ids, scores):
                    l_dict = {}
                    l_dict['category'] = class_names[int(class_id)]
                    x1, y1, w_, h_ = tlwh
                    #print(x1, y1,w_,h__)
                    x1 -= padding[0]
                    y1 -= padding[1]
                    x1 *= im_w / (bbox_w - 2 * padding[0])
                    w_  *= im_w / (bbox_w - 2 * padding[0])
                    y1 *= im_h / (bbox_h - 2 * padding[1])
                    h_  *= im_h / (bbox_h - 2 * padding[1])
                    if x1<0:
                        x1=0
                    if y1<0:
                        y1=0
                    if x1+w_>im_w:
                        w_=im_w-x1
                    if y1+h_>im_h:
                        h_=im_h-y1
                    line = save_format.format(
                                frame=frame-1, id=track_id,
                                x1=x1, y1=y1, w=w_, h=h_)
                    file.write(line)

    logger.info('save results to {}'.format(filename))
    
def format_dets_dict2dets_list(dets_dict, w, h):
    """
    :param dets_dict:
    :param w: input image width
    :param h: input image height
    :return:
    """
    dets_list = []
    for k, v in dets_dict.items():
        for det_obj in v:
            x1, y1, x2, y2, score, cls_id = det_obj
            center_x = (x1 + x2) * 0.5 / float(w)
            center_y = (y1 + y2) * 0.5 / float(h)
            bbox_w = (x2 - x1) / float(w)
            bbox_h = (y2 - y1) / float(h)

            dets_list.append([int(cls_id), score, center_x, center_y, bbox_w, bbox_h])

    return dets_list

def eval_seq(opt,
             data_loader,
             write_result,
             result_f_name,
             save_dir=None,
             show_image=True,
             frame_rate=30,
             mode='track',
             data_type='mot'):
    """
    :param opt:
    :param data_loader:
    :param data_type:
    :param result_f_name:
    :param save_dir:
    :param show_image:
    :param frame_rate:
    :param mode: track or detect
    :return:
    """
    if save_dir:
        mkdir_if_missing(save_dir)

    # tracker = JDETracker(opt, frame_rate)
    # tracker = YOLOBYTETracker(opt)
    tracker = YOLOTracker(opt)

    timer = Timer()

    results_dict = defaultdict(list)

    frame_id = 1  # frame index
    for path, img, img0, (dw, dh) in data_loader:
        # if frame_id % 30 == 0 and frame_id != 0:
            # logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1.0 / max(1e-5, timer.average_time)))

        # --- run tracking
        blob = torch.from_numpy(img).unsqueeze(0).to(opt.device)

        if mode == 'track':  # process tracking
            # ----- track updates of each frame
            timer.tic()

            online_targets_dict = tracker.update_tracking(blob, img0)

            timer.toc()
            # -----

            # collect current frame's result
            online_tlwhs_dict = defaultdict(list)
            online_ids_dict = defaultdict(list)
            online_scores_dict = defaultdict(list)
            for cls_id in range(opt.num_classes):  # process each class id
                online_targets = online_targets_dict[cls_id]
                for track in online_targets:
                    tlwh = track.tlwh
                    t_id = track.track_id
                    score = track.score
                    if tlwh[2] * tlwh[3] > opt.min_box_area:  # and not vertical:
                        online_tlwhs_dict[cls_id].append(tlwh)
                        online_ids_dict[cls_id].append(t_id)
                        online_scores_dict[cls_id].append(score)

            # collect result
            for cls_id in range(opt.num_classes):
                results_dict[cls_id].append((frame_id + 1,
                                             online_tlwhs_dict[cls_id],
                                             online_ids_dict[cls_id],
                                             online_scores_dict[cls_id]))

            # draw track/detection
            if show_image or save_dir is not None:
                if frame_id > 0:
                    online_im: ndarray = vis.plot_tracks(image=img0,
                                                         bbox_dim=(576, 1024),
                                                         padding=(dw, dh),
                                                         tlwhs_dict=online_tlwhs_dict,
                                                         obj_ids_dict=online_ids_dict,
                                                         num_classes=opt.num_classes,
                                                         frame_id=frame_id,
                                                         fps=1.0 / timer.average_time)

        if frame_id > 0:
            if show_image:
                cv2.imshow('online_im', online_im)
                vid_h, vid_w, _ = online_im.shape
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)       
        # update frame id
        frame_id += 1

    # write track/detection results
    if write_result:
        write_results(result_f_name,results_dict, 'mot', img_dim=(1080,1920),padding=(dw, dh))

    return frame_id, timer.average_time, timer.calls


def main(opt,
         data_root='/home/fatih/phd/FairCenterMOT/src/data',
         det_root=None, seqs=('SOMPT22'),
         exp_name='demo',
         save_images=False,
         save_videos=False,
         show_image=True):

    logger.setLevel(logging.INFO)
    
    epochnum = int(opt.load_model.split("_")[-1][:-4])

    
    exp_root = f"/home/fatih/phd/FairCenterMOT/exp/val/{exp_name}"
    result_root = f"/home/fatih/phd/FairCenterMOT/exp/val/{exp_name}/{epochnum}/"
    mkdir_if_missing(exp_root)
    mkdir_if_missing(result_root)
    

    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    
    for seq in tqdm(seqs):
        
        if seq in ".DS_Store":
            continue        
        
        output_dir = osp.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        
        dataloader = datasets.LoadImages(osp.join(data_root, seq))

        result_filename = osp.join(result_root, '{}.json'.format(seq))

        frame_rate = 30
        
        nf, ta, tc = eval_seq(opt, dataloader, True, result_filename,
                              save_dir=output_dir, show_image=show_image,
                              frame_rate=frame_rate, data_type=data_type)
        
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)
        
        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))    
    # ----- timing
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))
    
    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    opt = opts().init()
    opt.device = 0
    
    val_data = "/home/fatih/phd/FairCenterMOT/src/data/SOMPT22/train/SOMPT22-04/img1"
    seqs = os.listdir(val_data)

    main(opt,
         data_root=val_data,
         seqs=seqs,
         exp_name=opt.exp_id,
         show_image=False,
         save_images=False,
         save_videos=False)
