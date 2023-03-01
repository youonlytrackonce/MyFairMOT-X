from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic experiment setting
        self.parser.add_argument('task', default='mot', help='mot')
        self.parser.add_argument('--dataset', default='jde', help='jde')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--val', action='store_true')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--test_emb', action='store_true')
        self.parser.add_argument('--test_det', action='store_true')
        self.parser.add_argument('--start_epoch',
                                 type=int,
                                 default=1,
                                 help='epoch to start from for validation')
        self.parser.add_argument('--load_model', default='', help='path to pretrained model')
        self.parser.add_argument('--resume',
                                 action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # system
        self.parser.add_argument('--gpus',
                                 default='0',  # 0, 5, 6
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers',
                                 type=int,
                                 default=4,  # 8, 6, 4
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet
        self.parser.add_argument('--gen-scale',
                                 type=bool,
                                 default=True,
                                 help='Whether to generate multi-scales')
        self.parser.add_argument('--is_debug',
                                 type=bool,
                                 default=False,  # 是否使用多线程加载数据, default: False
                                 help='whether in debug mode or not')  # debug模式下只能使用单进程

        # log
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--hide_data_time', action='store_true',
                                 help='not display time during training.')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.5,
                                 help='visualization threshold.')

        # model: backbone and so on...
        self.parser.add_argument('--arch',
                                 default='yolox',
                                 help='model architecture. Currently tested'
                                      'resdcn_18 |resdcn_34 | resdcn_50 | resfpndcn_34 |'
                                      'dla_34 | hrnet_32 | hrnet_18 | cspdarknet_53 | regnet | effdet')
        self.parser.add_argument('--yolo',
                            default="l",
                            help='YOLO-X scale: S, M, L, X, etc.')
        self.parser.add_argument('--yolo_depth',
                                 default=-1,
                                 type=float,
                                 help='YOLO-X depth scaling for S, M, L, X, etc.')
        self.parser.add_argument('--yolo_width',
                                 default=-1,
                                 type=float,
                                 help='YOLO-X width scaling for S, M, L, X, etc.')
        self.parser.add_argument('--head_conv',
                                 type=int,
                                 default=-1,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '256 for resnets and 256 for dla.')
        self.parser.add_argument('--down_ratio',
                                 type=int,
                                 default=8,
                                 help='output stride. Currently only supports 8.')

        # input
        self.parser.add_argument('--input_res',
                                 type=int,
                                 default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h',
                                 type=int,
                                 default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w',
                                 type=int,
                                 default=-1,
                                 help='input width. -1 for default from dataset.')

        # train
        self.parser.add_argument('--lr',
                                 type=float,
                                 default=7e-5,  # 1e-4, 7e-5, 5e-5, 3e-5
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step',
                                 type=str,
                                 default='20,35,40,50,60,75,80',  # 20,27
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs',
                                 type=int,
                                 default=20,  # 30, 10, 3, 1
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size',
                                 type=int,
                                 default=16,  # 18, 16, 14, 12, 10, 8, 4
                                 help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu.')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=10,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--trainval',
                                 action='store_true',
                                 help='include validation in training and '
                                      'test on test set')
        
        # training augmentation
        self.parser.add_argument('--augment',
                                 action='store_true',
                                 help='whether to augment dataset')
        self.parser.add_argument('--mosaic',
                                 action='store_true',
                                 help='whether to use mosaic for augmentation')

        # demo
        self.parser.add_argument('--input_video', default="", help='video to demo')
        self.parser.add_argument('--output_format', default="video", help='demo output format')

        # test
        self.parser.add_argument('--K',
                                 type=int,
                                 default=200,  # 128
                                 help='max number of output objects.')  # 一张图输出检测目标最大数量
        self.parser.add_argument('--not_prefetch_test',
                                 action='store_true',
                                 help='not use parallal data pre-processing.')
        self.parser.add_argument('--fix_res',
                                 action='store_true',
                                 help='fix testing resolution or keep '
                                      'the original resolution')
        self.parser.add_argument('--keep_res',
                                 action='store_true',
                                 help='keep the original resolution'
                                      ' during validation.')
        self.parser.add_argument('--kitti_test',
                                 action='store_true')

        # tracking
        self.parser.add_argument(
            '--test_mot16', default=False, help='test mot16')
        self.parser.add_argument(
            '--val_mot15', default=False, help='val mot15')
        self.parser.add_argument(
            '--test_mot15', default=False, help='test mot15')
        self.parser.add_argument(
            '--val_mot16', default=False, help='val mot16 or mot15')
        self.parser.add_argument(
            '--test_mot17', default=False, help='test mot17')
        self.parser.add_argument(
            '--val_mot17', default=False, help='val mot17')
        self.parser.add_argument(
            '--val_mot20', default=False, help='val mot20')
        self.parser.add_argument(
            '--test_mot20', default=False, help='test mot20')
            
        # confidence
        self.parser.add_argument('--conf_thre',
                                type=float,
                                default=0.4,  # 0.6, 0.4
                                help='confidence thresh for tracking')
        self.parser.add_argument('--det_thre',
                                 type=float,
                                 default=0.4,
                                 help='confidence thresh for detection')
        self.parser.add_argument('--nms_thre',
                                 type=float,
                                 default=0.4,
                                 help='iou thresh for nms')
        self.parser.add_argument('--track_buffer',
                                 type=int,
                                 default=30,  # 30
                                 help='tracking buffer')
        self.parser.add_argument('--min-box-area',
                                 type=float,
                                 default=20,
                                 help='filter out tiny boxes')

        # 测试阶段的输入数据模式: video or image dir
        self.parser.add_argument('--input-mode',
                                 type=str,
                                 default='video',  # video or image_dir or img_path_list_txt
                                 help='input data type(video or image dir)')

        # 输入的image目录
        self.parser.add_argument('--input-img',
                                 type=str,
                                 default='/users/duanyou/c5/all_pretrain/test.txt',  # ../images/
                                 help='path to the input image directory or image file list(.txt)')

        self.parser.add_argument('--output-root',
                                 type=str,
                                 default='../results',
                                 help='expected output root path')

        # dataset
        self.parser.add_argument('--data_cfg', type=str,
                                 default='/home/fatih/phd/FairCenterMOT/src/lib/cfg/mot20-sompt22.json',  # mcmot.json, mcmot_det.json,
                                 help='load data from cfg')
        self.parser.add_argument('--data_dir',
                                 type=str,
                                 default='/home/fatih/phd/FairCenterMOT/src/data/')

        # loss
        self.parser.add_argument('--uncertainty_loss',
                                 dest='uncertainty_loss',
                                 action='store_true')
        self.parser.add_argument('--no_uncertainty_loss',
                                 dest='uncertainty_loss',
                                 action='store_false')
        self.parser.set_defaults(uncertainty_loss=True)
        
        self.parser.add_argument('--detection_only',
                                 dest='detection_only',
                                 action='store_true')
        
        self.parser.add_argument('--reid_only',
                                 action='store_true')
        
        self.parser.add_argument('--freeze_backbone',
                                 action='store_true')
        
        self.parser.add_argument('--l1_loss',
                                 action='store_true')
        
        self.parser.add_argument('--reg_loss',
                                 default='l1',
                                 help='regression loss: sl1 | l1 | l2')  # sl1: smooth L1 loss
        self.parser.add_argument('--hm_weight',
                                 type=float,
                                 default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight',
                                 type=float,
                                 default=1,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight',
                                 type=float,
                                 default=0.1,
                                 help='loss weight for bounding box size.')
        self.parser.add_argument('--id_loss',
                                 default='ce',
                                 help='reid loss: ce | triplet')
        self.parser.add_argument('--id_weight',
                                 type=float,
                                 default=1,  # 0 for detection only and 1 for detection and re-id
                                 help='loss weight for id')  # ReID feature extraction or not
        self.parser.add_argument('--reid_dim',
                                 type=int,
                                 default=128,  # 128, 256, 512
                                 help='feature dim for reid')
        self.parser.add_argument('--post_conv_layers',
                                 type=int,
                                 default=0,  # 128, 256, 512
                                 help='num conv2d layers after feature fusion for reid')
        self.parser.add_argument('--input_wh',
                                 type=tuple,
                                 default=(1024, 576),  # (768, 448) or (1088, 608)
                                 help='net input resplution')
        self.parser.add_argument('--multi_scale',
                                 type=bool,
                                 default=False,
                                 help='Whether to use multi-scale training or not')

        # ---------------------- Visdrone Dataset Classes ----------------------
        # pedestrian      (1),  --> 0 
        # people          (2),  --> 1
        # bicycle         (3),  --> 2
        # car             (4),  --> 3
        # van             (5),  --> 4
        # truck           (6),  --> 5
        # tricycle        (7),  --> 6
        # awning-tricycle (8),  --> 7
        # bus             (9),  --> 8
        # motor           (10), --> 9
        # -----------------------------------------------------------------------
        
        # ----------------------- BDD100K Dataset Classes -----------------------
        # pedestrian      (1),  --> 0 
        # rider           (2),  --> 1
        # car             (3),  --> 2
        # truck           (4),  --> 3
        # bus             (5),  --> 4
        # train           (6),  --> 5
        # motorcycle      (7),  --> 6
        # bicycle         (8),  --> 7
        # -----------------------------------------------------------------------


        self.parser.add_argument('--reid_cls_ids',
                                 default='0,1,2,3,4,5,6,7',
                                 help='')  # the object classes need to do reid

        self.parser.add_argument('--norm_wh', action='store_true',
                                 help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
        self.parser.add_argument('--dense_wh', action='store_true',
                                 help='apply weighted regression near center or '
                                      'just apply regression on center point.')
        self.parser.add_argument('--cat_spec_wh',
                                 action='store_true',
                                 help='category specific bounding box size.')
        self.parser.add_argument('--not_reg_offset',
                                 action='store_true',
                                 help='not regress local offset.')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        opt.fix_res = not opt.keep_res

        opt.reg_offset = not opt.not_reg_offset

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 256
        opt.pad = 31
        opt.num_stacks = 1

        if opt.trainval:
            opt.val_intervals = 100000000

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        print('Training Chunk Sizes:', opt.chunk_sizes)

        opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
            model_path = os.path.join('/home/fatih/phd/FairCenterMOT/models/', opt.exp_id)
            opt.load_model = os.path.join(model_path, 'model_last.pth')

            if not os.path.exists(opt.load_model):
                print("No last model to load in modelrepo. Trying opt.save_dir.")
                
                opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')
                
                if not os.path.exists(opt.load_model):
                    print("Although --resume was specified, there is no model to load. Training from Epoch 1.")
                    opt.load_model = ''
                else:
                    print("Loaded model from opt.save_dir. Next model will be saved to modelrepo.")
        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        """
        :param opt:
        :param dataset:
        :return:
        """
        
        opt.num_classes = 1
        
        yolo_dict = {
            "s" : (0.33, 0.50),
            "m" : (0.67, 0.75),
            "l" : (1.00, 1.00),
            "x" : (1.33, 1.25)
        }
        
        if opt.yolo_depth == -1 or opt.yolo_width == -1:
            print(f"Using Model Scale for YOLOX-{opt.yolo.upper()}")
            opt.yolo_depth, opt.yolo_width = yolo_dict[opt.yolo]
        

        for reid_id in opt.reid_cls_ids.split(','):
            if int(reid_id) > opt.num_classes - 1:
                print('[ERROR]: Configuration conflict of reid_cls_ids and num_classes!')
                return
            
        opt.nID_dict = dataset.nID_dict

        print("Heads are Predefined in YOLOX!")

        return opt
        
    def init(self, args=''):
            opt = self.parse(args)

            default_dataset_info = {
                'mot': {'default_input_wh': [1024, 576],
                        'num_classes': len(opt.reid_cls_ids.split(',')),  # 1
                        'mean': [0.408, 0.447, 0.470],
                        'std': [0.289, 0.274, 0.278],
                        # 'dataset': 'bdd100k',
                        # 'nID': 14455,
                        'nID_dict': {}},
            }

            class Struct:
                def __init__(self, entries):
                    for k, v in entries.items():
                        self.__setattr__(k, v)
            dataset = Struct(default_dataset_info[opt.task])
            # opt.dataset = dataset.dataset
            opt = self.update_dataset_info_and_set_heads(opt, dataset)

            return opt
