from gettext import translation
import numpy as np
import itertools
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys, os
import errno
import copy
import time
import math
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
from tensorboardX import SummaryWriter

from common.arguments import parse_args
from common.utils import deterministic_random, save_model, save_model_epoch
from common.camera import *
from common.multiview_model import get_models
from common.loss import *
from common.generators import *
from common.data_augmentation_multi_view import *
from common.h36m_dataset import Human36mCamera, Human36mDataset, Human36mCamDataset
from common.set_seed import *
from common.config import config as cfg
from common.config import reset_config, update_config
from common.vis import *
dataset_path = '../MHFormer/dataset/data_3d_h36m.npz'
set_seed()

args = parse_args()
update_config(args.cfg) ###config file->cfg
reset_config(cfg, args) ###arg -> cfg
print(cfg)
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cfg.GPU)

print('p2d detector:{}'.format('gt_p2d' if cfg.DATA.USE_GT_2D else cfg.H36M_DATA.P2D_DETECTOR))
HumanCam = Human36mCamera(cfg)

keypoints = {}
for sub in [1, 5, 6, 7, 8, 9, 11]:
    if cfg.H36M_DATA.P2D_DETECTOR == 'cpn' or cfg.H36M_DATA.P2D_DETECTOR == 'gt':
        data_pth = 'data/h36m_sub{}.npz'.format(sub)
    elif cfg.H36M_DATA.P2D_DETECTOR == 'ada_fuse':
        data_pth = 'data/h36m_sub{}_ada_fuse.npz'.format(sub)
    
    keypoint = np.load(data_pth, allow_pickle=True)
    lst = keypoint.files
    keypoints_metadata = keypoint['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    keypoints['S{}'.format(sub)] = keypoint['positions_2d'].item()['S{}'.format(sub)]

kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = [kps_left, kps_right]

N_frame_action_dict = {
2699:'Directions',2356:'Directions',1552:'Directions',
5873:'Discussion', 5306:'Discussion',2684:'Discussion',2198:'Discussion',
2686:'Eating', 2663:'Eating',2203:'Eating',2275 :'Eating',
1447:'Greeting', 2711:'Greeting',1808:'Greeting', 1695:'Greeting',
3319:'Phoning',3821:'Phoning',3492:'Phoning',3390:'Phoning',
2346:'Photo',1449:'Photo',1990:'Photo',1545:'Photo',
1964:'Posing', 1968:'Posing',1407:'Posing',1481 :'Posing',
1529:'Purchases', 1226:'Purchases',1040:'Purchases', 1026:'Purchases',
2962:'Sitting', 3071:'Sitting',2179:'Sitting', 1857:'Sitting',
2932:'SittingDown', 1554:'SittingDown',1841:'SittingDown', 2004:'SittingDown',
4334:'Smoking',4377:'Smoking',2410:'Smoking',2767:'Smoking',
3312:'Waiting', 1612:'Waiting',2262:'Waiting', 2280:'Waiting',
2237:'WalkDog', 2217:'WalkDog',1435:'WalkDog', 1187:'WalkDog',
1703:'WalkTogether',1685:'WalkTogether',1360:'WalkTogether',1793:'WalkTogether',
1611:'Walking', 2446:'Walking',1621:'Walking', 1637:'Walking',
}

actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases','Sitting','SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking']
train_actions = actions
test_actions = actions
vis_actions = actions


action_frames = {}
for act in actions:
    action_frames[act] = 0
for k,v in N_frame_action_dict.items():
    action_frames[v] += k
if cfg.H36M_DATA.P2D_DETECTOR == 'cpn' or cfg.H36M_DATA.P2D_DETECTOR == 'gt':
    vis_score = pickle.load(open('./data/score.pkl', 'rb'))
elif cfg.H36M_DATA.P2D_DETECTOR[:3] == 'ada':
    vis_score = pickle.load(open('./data/vis_ada.pkl', 'rb'))

def fetch(subjects, action_filter=None,  parse_3d_poses=True, is_test = False, out_plus = False):
    out_poses_3d = []
    out_poses_2d_view1 = []
    out_poses_2d_view2 = []
    out_poses_2d_view3 = []
    out_poses_2d_view4 = []
    out_camera_params = []
    out_poses_3d = []
    out_subject_action = []
    used_cameras = cfg.H36M_DATA.TEST_CAMERAS if is_test else cfg.H36M_DATA.TRAIN_CAMERAS
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a) and len(action.split(a)[1]) <3:
                        found = True
                        break
                if not found:
                    continue
            
            poses_3d = HumanCam._data[subject][action]['positions']
            cam_info = HumanCam._data[subject][action]['cameras']
            poses_2d = keypoints[subject][action]
            out_subject_action.append([subject, action])
            n_frames = poses_2d[0].shape[0]
            vis_name_1 = '{}_{}.{}'.format(subject, action, 0)
            vis_name_2 = '{}_{}.{}'.format(subject, action, 1)
            vis_name_3 = '{}_{}.{}'.format(subject, action, 2)
            vis_name_4 = '{}_{}.{}'.format(subject, action, 3)
            vis_score_cam0 = vis_score[vis_name_1][:n_frames][...,np.newaxis]
            vis_score_cam1 = vis_score[vis_name_2][:n_frames][...,np.newaxis]
            vis_score_cam2 = vis_score[vis_name_3][:n_frames][...,np.newaxis]
            vis_score_cam3 = vis_score[vis_name_4][:n_frames][...,np.newaxis]
            if vis_score_cam3.shape[0] != vis_score_cam2.shape[0]:
                vis_score_cam2 = vis_score_cam2[:-1]
                vis_score_cam1 = vis_score_cam1[:-1]
                vis_score_cam0 = vis_score_cam0[:-1]
                for i in range(4):
                    poses_2d[i] = poses_2d[i][:-1]
                    
            if is_test == True and action == 'Walking' and poses_2d[0].shape[0] == 1612:
                out_poses_2d_view1.append(np.concatenate((poses_2d[0][1:], vis_score_cam0[1:]), axis =-1))
                out_poses_2d_view2.append(np.concatenate((poses_2d[1][1:], vis_score_cam1[1:]), axis =-1))
                out_poses_2d_view3.append(np.concatenate((poses_2d[2][1:], vis_score_cam2[1:]), axis =-1))
                out_poses_2d_view4.append(np.concatenate((poses_2d[3][1:], vis_score_cam3[1:]), axis =-1))
            else:
                out_poses_2d_view1.append(np.concatenate((poses_2d[0], vis_score_cam0), axis =-1))
                out_poses_2d_view2.append(np.concatenate((poses_2d[1], vis_score_cam1), axis =-1))
                out_poses_2d_view3.append(np.concatenate((poses_2d[2], vis_score_cam2), axis =-1))
                out_poses_2d_view4.append(np.concatenate((poses_2d[3], vis_score_cam3), axis =-1))
            out_poses_3d.append(poses_3d)

    
    final_pose = []
    if 0 in used_cameras:
        final_pose.append(out_poses_2d_view1)
    if 1 in used_cameras:
        final_pose.append(out_poses_2d_view2)
    if 2 in used_cameras:
        final_pose.append(out_poses_2d_view3)
    if 3 in used_cameras:
        final_pose.append(out_poses_2d_view4)
        
    
    if is_test is True:
        if out_plus == True:
            return final_pose, out_poses_3d
        else:
            return final_pose
    else:
        if out_plus == True:
            return final_pose, out_subject_action, out_poses_3d
        else:
            return final_pose, out_subject_action

use_2d_gt = cfg.DATA.USE_GT_2D
receptive_field = cfg.NETWORK.TEMPORAL_LENGTH
pad = receptive_field // 2
causal_shift = 0
model, model_test = get_models(cfg)

#####模型参数量、计算量(MACs)、inference time
if cfg.VIS.VIS_COMPLEXITY:
    from thop import profile
    from thop import clever_format
    if args.eval:
        from ptflops import get_model_complexity_info
    #####模型参数量、计算量(MACs)
    receptive_field = 1
    model_test.eval()
    for i in range(1,5):
        input = torch.randn(1, receptive_field,17,3,i)
        rotation = torch.randn(1, 3, 3,receptive_field,i,i)
        macs, params = profile(model_test, inputs=(input, rotation))
        macs, params = clever_format([macs, params], "%.3f")
        print('view: {} T: {} MACs:{} params:{}'.format(i, receptive_field, macs, params))
        if args.eval:
            flops, params = get_model_complexity_info(model_test, (receptive_field,17,3,i), as_strings=True, print_per_layer_stat=False)
            print('Flops:{}, Params:{}'.format(flops, params))
    #####inference time
    infer_model = model_test.cuda()
    infer_model.eval()
    for receptive_field in [1, 27]:
        for i in range(1,5):
            input = torch.randn(1, receptive_field,17,3,i).float().cuda()
            rotation = torch.randn(1, 3, 3,receptive_field,i,i).float().cuda()
            
            for k in range(100):
                out = infer_model(input, rotation)
            
            N = 1000
            torch.cuda.synchronize()
            start_time = time.time()
            for n in range(N):
                infer_model(input, rotation)
            torch.cuda.synchronize()
            end_time = time.time()
            print('n_frames:{} n_views: {}  time:{:.4f}'.format(receptive_field, i, (end_time - start_time) / N))
    exit()
else:
    total_params = sum(p.numel() for p in model_test.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model_test.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

EVAL = args.eval
ax_views = []

if EVAL and cfg.VIS.VIS_3D:
    plt.ion()
    vis_tool = Vis(cfg, 2)
        
def load_state(model_train, model_test):
    train_state = model_train.state_dict()
    test_state = model_test.state_dict()
    pretrained_dict = {k:v for k, v in train_state.items() if k in test_state}
    test_state.update(pretrained_dict)
    model_test.load_state_dict(test_state)
    
if EVAL and not cfg.TEST.TRIANGULATE:
    chk_filename = cfg.TEST.CHECKPOINT
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_checkpoint = checkpoint['model'] if 'best_model' not in checkpoint.keys() else checkpoint['best_model']
    train_checkpoint = model.state_dict()
    test_checkpoint = model_test.state_dict()
    for k, v in train_checkpoint.items():
        if k not in model_checkpoint.keys():
            continue
        checkpoint_v = model_checkpoint[k]
        if 'p_shrink.shrink' in k:
            if model_checkpoint[k].shape[0] == 32:
                checkpoint_v = checkpoint_v[1::2]

        train_checkpoint[k] = checkpoint_v

    print('EVAL: This model was trained for {} epochs'.format(checkpoint['epoch']))
    model.load_state_dict(train_checkpoint)

if True:
    if not cfg.DEBUG and (not args.eval or args.log):
        summary_writer = SummaryWriter(log_dir=cfg.LOG_DIR)
    else:
        summary_writer = None
    
    poses_train_2d, subject_action = fetch(cfg.H36M_DATA.SUBJECTS_TRAIN, train_actions)
    _, _, poses_3d_extra = fetch(cfg.H36M_DATA.SUBJECTS_TRAIN, train_actions, out_plus = True)

    lr = cfg.TRAIN.LEARNING_RATE
    if cfg.TRAIN.RESUME:
        chk_filename = cfg.TRAIN.RESUME_CHECKPOINT
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('RESUME: This model was trained for {} epochs'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['model'])
    if torch.cuda.is_available() and not cfg.TEST.TRIANGULATE:
        model = torch.nn.DataParallel(model).cuda()
        model_test = torch.nn.DataParallel(model_test).cuda()
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=lr, amsgrad=True)    
    if cfg.TRAIN.RESUME:
        epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_result_epoch = checkpoint['best_epoch']
        best_state_dict = checkpoint['best_model']
        lr = checkpoint['lr']
        best_result = 100
    else:
        epoch = 0
        best_result = 100
        best_state_dict = None
        best_result_epoch = 0
        
    lr_decay = cfg.TRAIN.LR_DECAY
    initial_momentum = 0.1
    final_momentum = 0.001
    train_generator = ChunkedGenerator(cfg.TRAIN.BATCH_SIZE, poses_train_2d, 1,pad=pad, causal_shift=causal_shift, shuffle=True, augment=False, kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right, sub_act=subject_action, extra_poses_3d=None) if cfg.H36M_DATA.PROJ_Frm_3DCAM == True else ChunkedGenerator(cfg.TRAIN.BATCH_SIZE, poses_train_2d, 1,pad=pad, causal_shift=causal_shift, shuffle=True, augment=True,kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
 
    print('** Starting.')
    
    data_aug = DataAug(cfg, add_view = cfg.TRAIN.NUM_AUGMENT_VIEWS)
    iters = 0
    msefun = torch.nn.L1Loss() 
    num_train_views = len(cfg.H36M_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS

    # while epoch < cfg.TRAIN.NUM_EPOCHES:
    start_time = time.time()
    model.train()
    process = tqdm(total = train_generator.num_batches)
    uncertainty = torch.zeros(cfg.TRAIN.BATCH_SIZE, 17, 2, 4)
    vis_all = torch.zeros(cfg.TRAIN.BATCH_SIZE, 17, 4)
    for batch_2d, sub_action, batch_flip in train_generator.next_epoch():
        if EVAL:
            break
        process.update(1)
        inputs = torch.from_numpy(batch_2d.astype('float32'))
        assert inputs.shape[-2] == 8 #(p2d_gt, p2d_pre, p3d, vis)
        inputs_2d_gt = inputs[...,:,:2,:]
        inputs_2d_pre = inputs[...,2:4,:]
        cam_3d = inputs[..., 4:7,:]
        B, T, V, C, N = cam_3d.shape
        if use_2d_gt:
            vis = torch.ones(B, T, V, 1, N)
        else:
            vis = inputs[...,7:8, :]

        if cfg.TRAIN.NUM_AUGMENT_VIEWS:
            vis = torch.cat((vis, torch.ones(B, T, V, 1, cfg.TRAIN.NUM_AUGMENT_VIEWS)), dim = -1)


        inputs_3d_gt = cam_3d.cuda()
        view_list = list(range(num_train_views))

        if cfg.TRAIN.NUM_AUGMENT_VIEWS > 0:
            pos_gt_3d_tmp = copy.deepcopy(inputs_3d_gt)
            pos_gt_2d, pos_gt_3d = data_aug(pos_gt_2d = inputs_2d_gt, pos_gt_3d = pos_gt_3d_tmp)
            pos_pre_2d = torch.cat((inputs_2d_pre, pos_gt_2d[...,inputs_2d_pre.shape[-1]:]), dim = -1)

            if use_2d_gt:
                h36_inp = pos_gt_2d[..., view_list]
            else:
                h36_inp = pos_pre_2d[..., view_list]
            pos_gt = pos_gt_3d[..., view_list]

        else:
            if use_2d_gt:
                h36_inp = inputs_2d_gt[..., view_list]
            else:
                h36_inp = inputs_2d_pre[..., view_list]
            pos_gt = inputs_3d_gt[..., view_list]

        p3d_gt_ori = copy.deepcopy(pos_gt)
        p3d_root = copy.deepcopy(pos_gt[:,:,:1]) #(B,T, 1, 3, N)
        vis = vis[...,:4].squeeze()
        pos_gt_2d = pos_gt_2d[...,:4].squeeze()
        pos_pre_2d = pos_pre_2d[...,:4].squeeze()
        uncertainty_tmp = abs(pos_gt_2d-pos_pre_2d)
        uncertainty = torch.cat([uncertainty, uncertainty_tmp], dim=0)
        vis_all = torch.cat([vis_all, vis], dim=0)
    uncertainty = uncertainty[720:,...].numpy()
    vis_all = vis_all[720:,...].numpy()
    np.save("uncertainty_2d_pre_gt.npz", uncertainty)
    np.save("vis_all.npz", vis_all)
    
    assert uncertainty.shape[0]==vis_all.shape[0]
    
    print("1111111111111111111111")
            
            