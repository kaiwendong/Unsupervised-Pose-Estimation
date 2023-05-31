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
#dataset_path = '../MHFormer/dataset/data_3d_h36m.npz'
dataset_path = "./data/data_3d_h36m.npz"
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

#HumanData = Human36mDataset(cfg, keypoints)
#HumanCamData = Human36mCamDataset(keypoints)

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
    if model_checkpoint == None:
        model_checkpoint = checkpoint['model']
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

    while epoch < cfg.TRAIN.NUM_EPOCHES:

        start_time = time.time()
        model.train()
        process = tqdm(total = train_generator.num_batches)
        # idx = 0
        for batch_2d, sub_action, batch_flip in train_generator.next_epoch():
            # if idx > 2:
            #     break
            # idx += 1
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
            #if cfg.H36M_DATA.PROJ_Frm_3DCAM == True:
            #    prj_3dgt_to_2d= HumanCam.p3d_im2d(pos_gt, sub_action, view_list)
            p3d_gt_ori = copy.deepcopy(pos_gt)
            p3d_root = copy.deepcopy(pos_gt[:,:,:1]) #(B,T, 1, 3, N)
            pos_gt[:,:,:1] = 0
            # print("rel, view 0 : "+str(pos_gt[0,0,:, :, 0]))
            # print("rel, view 1 : "+str(pos_gt[0, 0, :, :, 1]))
            p3d_gt_abs = pos_gt+p3d_root
            # print("abs, view 0 : "+str(p3d_gt_abs[0,0,:, :, 0]))
            # print("abs, view 1 : " + str(p3d_gt_abs[0, 0, :, :, 1]))
            # input()
            optimizer.zero_grad()
            inp = torch.cat((h36_inp, vis), dim = -2)
            if cfg.NETWORK.USE_GT_TRANSFORM or cfg.TRAIN.USE_ROT_LOSS:
                #相机之间的旋转
                rotation = get_rotation(pos_gt[:,:1]) #(B, 3, 3, 1, N, N)

                # #相机之间的位移
                # #print(rotation)
                # t = torch.einsum('btjcn,bqcmn->btjqmn', p3d_root[:,:1], rotation[:,:,:,0])#(B, T, 1, 3, N, N)
                # t = t - t[...,:1]
                # t = t.permute(0, 2, 3, 1, 4, 5) #(B, 1, 3, T, N, N)
                # if cfg.NETWORK.M_FORMER.GT_TRANSFORM_MODE == 'rt':
                #     rotation = torch.cat((rotation, t), dim = 1)
            else:
                rotation = None
 
            if cfg.TRAIN.USE_INTER_LOSS:
                print('Input shape is {}'.format(inp.shape))
                if cfg.TRAIN.TEMPORAL_SMOOTH_LOSS_WEIGHT is not None:
                    out, out_full, other_out, tran, pred_rot = model(inp, rotation)
                else:
                    out, other_out, tran, pred_rot = model(inp, rotation) #mask:(B, 1, 1, 1, N, N)
            else:
                out = model(inp, rotation)
           
            # if cfg.TRAIN.PRJ_2DIM_TO_3DWD:
            #     prj_2dpre_to_3d = HumanCam.p2d_cam3d_batch(h36_inp[:, pad:pad+1, :, :, :4], sub_action, view_list[:4], debug=False)
            #     B, _, J, _, _ = h36_inp.shape
            #     p_2dgt_to_3dcam = torch.zeros(B, 1, J, 3, 4)
            #     p_2dgt_to_3dwd  = torch.zeros(B, 1, J, 3)
            #     for inx, sub_ac in enumerate(sub_action):
            #         p_2dgt_to_3dcam[inx], p_2dgt_to_3dwd[inx]  =  HumanCam.p2d_cam3d(inp[inx:inx+1, pad:pad+1, :, :2, :4], sub_ac[0], view_list[:4], debug=True)
            #     prj_3dcam_to_3dwd = HumanCam.p3dcam_3dwd_batch(prj_2dpre_to_3d, sub_action, view_list[:4])
            #     prj_2dim_3dwd = HumanCam.p2d_world3d_batch(h36_inp[:,pad:pad+1], sub_action, view_list).to(out.device)
            #     prj_2dim_3dwd_gt = HumanCam.p2d_world3d_batch(pos_gt_2d, sub_action, view_list).to(out.device)
            #     pri_3dcam_gt_to_3dwd = HumanCam.p3dcam_3dwd_batch(p3d_gt_abs[:, pad:pad+1], sub_action, view_list)
            #     pose_3d_gt_extra = torch.from_numpy(batch_3d[:,pad:pad+1,:,:]).to(out.device)
            #     prj_3dwd_to_3dcam_gt = HumanCam.p3dwd_p3dcam_batch(pose_3d_gt_extra.squeeze(), sub_action, view_list)
            #     print(mpjpe(pos_gt[:,pad:pad+1,...,:4].squeeze().permute(0,3,1,2), prj_3dwd_to_3dcam_gt))
            #     loss_wdgt1 = msefun(pose_3d_gt_extra, pri_3dcam_gt_to_3dwd[0,0])
            #     #loss_wdgt2 = msefun(pose_3d_gt_extra, prj_2dim_3dwd_gt[0,pad:pad+1].to(out.device))
            #     triangu_loss = mpjpe(prj_2dpre_to_3d.to(out.device), out[...,:4])
            #     if summary_writer is not None:
            #         summary_writer.add_scalar("triangu_loss/iter", triangu_loss, iters)
            # else:
            #     triangu_loss = 0
            # print('triangu_loss is {}'.format(triangu_loss))

            #out = torch.mean(out, -1, keepdim=True).repeat(1, 1, 1, 1, 4)
            out = torch.mean(out, -1)
            # if cfg.H36M_DATA.PROJ_Frm_3DCAM == True:
            #     #data_gt_extra = np.load('../MHFormer/dataset/data_3d_h36m.npz', allow_pickle=True)
            #     prj_3dpre_to_2d = HumanCam.p3d_im2d_batch(out, sub_action, view_list, with_distor=True, flip=batch_flip)
            #     #if cfg.TRAIN.TEMPORAL_SMOOTH_LOSS_WEIGHT is not None:
            #     #prj_3dpre_to_2d_full = HumanCam.p3d_im2d_batch(out_full+p3d_root, sub_action, view_list, with_distor=True)
            #     prj_3dgt_abs_to_2d = HumanCam.p3d_im2d_batch(p3d_gt_abs[:, pad:pad+1], sub_action, view_list, with_distor=True, flip=batch_flip, gt_2d=inputs_2d_gt[:, pad:pad+1, :, :].to(out.device))
            #     prj_2dgt_to_3d = HumanCam.p2d_cam3d_batch(inputs_2d_gt[:, pad:pad+1, :, :], sub_action, view_list[:4], debug=False)
            #     prj_2dpre_to_3d = HumanCam.p2d_cam3d_batch(h36_inp[:, pad:pad+1, :, :, :4], sub_action, view_list[:4], debug=False)
            #     # print(mpjpe(pos_gt[:, pad:pad+1,...,:4], prj_2dgt_to_3d.to(out.device)), mpjpe(prj_3dgt_abs_to_2d.permute(0,2,3,4,1), inputs_2d_gt[:, pad:pad+1, :, :].to(out.device)))
            #     # print(mpjpe(prj_3dgt_abs_to_2d.permute(0, 2, 3, 4, 1), inputs_2d_gt[:, pad:pad + 1, :, :].to(out.device)))
            #     # print("prj_3dgt_abs_to_2d.permute(0, 2, 3, 4, 1).shape : "+str(prj_3dgt_abs_to_2d.permute(0, 2, 3, 4, 1).shape ))
            #     # print("inputs_2d_gt[:, pad:pad + 1, :, :].shape  : "+str(inputs_2d_gt[:, pad:pad + 1, :, :].shape ))
            #     # for v in range(4):
            #     #     h_inp_2d = inputs_2d_gt[:, pad:pad + 1, :, :][0, 0, :, 0, v].detach().numpy()
            #     #     w_inp_2d = inputs_2d_gt[:, pad:pad + 1, :, :][0, 0, :, 1, v].detach().numpy()
            #     #     h_3D_to_2D = prj_3dgt_abs_to_2d.permute(0, 2, 3, 4, 1).detach().cpu().numpy()[0, 0, :, 0, v]
            #     #     w_3D_to_2D = prj_3dgt_abs_to_2d.permute(0, 2, 3, 4, 1).detach().cpu().numpy()[0, 0, :, 1, v]
            #     #     plt.figure("view : "+str(v))
            #     #     plt.scatter(h_inp_2d, w_inp_2d, color='r', marker='o', label="input_2d")
            #     #     plt.scatter(h_3D_to_2D, w_3D_to_2D, color='g', marker='+', label="prj_3dgt_abs_to_2d")
            #     #     plt.legend()
            #     # plt.show()
            #print(out.shape)
            #pose_3D_in_cam_space = HumanCam.p3dwd_p3dcam_batch(out[:, 0, :, :, 0], sub_action, view_list)
            pose_3D_in_cam_space = HumanCam.p3dwd_p3dcam_batch(out[:, 0, :, :], sub_action, view_list)
            pose_3D_in_cam_space = pose_3D_in_cam_space.permute(0, 2, 3, 1)
            pose_3D_in_cam_space = torch.unsqueeze(pose_3D_in_cam_space, 1)
            # print("pose_3D_in_cam_space.shape : "+str(pose_3D_in_cam_space.shape))
            #input()
            pose_2D_from3D = HumanCam.p3d_im2d_batch(pose_3D_in_cam_space, sub_action, view_list, with_distor=True, flip=batch_flip)

            # print("pose_2D_from3D.shape : " + str(pose_2D_from3D.shape))
            pose_2D_from3D = pose_2D_from3D.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, N, J. C)
            # print("pose_2D_from3D.shape : " + str(pose_2D_from3D.shape))
            #prj_3dpre_to_2d = torch.where(prj_3dpre_to_2d < torch.tensor(1.).cuda(), prj_3dpre_to_2d, torch.tensor(1.).cuda())
            #prj_3dpre_to_2d = torch.where(prj_3dpre_to_2d > torch.tensor(-1.).cuda(), prj_3dpre_to_2d, torch.tensor(-1.).cuda())

            label_2D = h36_inp.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, N, J. C)
            # print("inputs_2d_gt.shape : " + str(inputs_2d_gt.shape))
            # print("prj_3dpre_to_2d.shape : " + str(prj_3dpre_to_2d.shape))
            # print("inputs_2d_gt[:, pad:pad + 1, :, :].shape : " + str(inputs_2d_gt[:, pad:pad + 1, :, :].shape))
            # input()
            # print(torch.min(inputs_2d_gt[:, pad:pad + 1, :, :]))
            # print(torch.max(inputs_2d_gt[:, pad:pad + 1, :, :]))
            # print(torch.min(prj_3dpre_to_2d))
            # print(torch.max(prj_3dpre_to_2d))
            # input()

            loss_consis_wd = mpjpe(pose_2D_from3D, label_2D[:, pad:pad + 1, :, :].to(out.device))

            #loss_consis_weight = cfg.TRAIN.CONSIS_LOSS_WEIGHT if cfg.TRAIN.VISI_WEIGHT==False else vis[:,pad:pad+1,:,:,:4].permute(0,4,1,2,3).to(out.device)

            # if cfg.TRAIN.PROJ_3DCAM_TO_3DWD:
            #     pri_3dcam_pre_to_3dwd = HumanCam.p3dcam_3dwd_batch(out, sub_action, view_list)
            #     pri_3dcam_gt_to_3dwd = HumanCam.p3dcam_3dwd_batch(p3d_gt_abs[:, pad:pad+1], sub_action, view_list)
            #     pair_loss = torch.zeros(6)
            #     for inx, _view_list in enumerate(itertools.combinations(list(range(len(cfg.H36M_DATA.TRAIN_CAMERAS))), 2)):
            #         pair_loss[inx] = msefun(pri_3dcam_pre_to_3dwd[:,_view_list[0]], pri_3dcam_pre_to_3dwd[:,view_list[1]])
            #     wd3d_pair_loss = pair_loss.mean()
            #     if cfg.TRAIN.PRJ_3DWD_TO_2DIM:
            #         p3dwd_pre_avg = torch.mean(pri_3dcam_pre_to_3dwd, dim=1) if not cfg.TRAIN.TAKE_OUT_AS_3DWD else torch.mean((out+p3d_root[:, pad:pad+1]).permute(0,4,1,2,3).squeeze()[:,cfg.H36M_DATA.TRAIN_CAMERAS], dim=1)
            #         #p3dwd_gt_avg = torch.mean(pri_3dcam_gt_to_3dwd, dim=1)
            #         #prj_3dwd_to_3dcam_pre = HumanCam.p3dwd_p3dcam_batch(p3dwd_pre_avg, sub_action, view_list)
            #         #prj_3dwd_to_3dcam_gt, prj_3dwd_to_2dim_gt = HumanCam.p3dwd_p3dcam_batch(p3dwd_gt_avg, sub_action, view_list, True)
            #         prj_3dwd_to_2dim_w_disr = HumanCam.p3d_im2d_batch(prj_3dwd_to_3dcam_pre.permute(0,2,3,1), sub_action, view_list, with_distor=True)
            #         #prj_3dwd_to_2dim_w_disr_gt = HumanCam.p3d_im2d_batch(prj_3dwd_to_3dcam_gt.permute(0,2,3,1), sub_action, view_list, with_distor=True)
            #         #prj_3dwd_to_2dim_gt = HumanCam.p3dwd_p2dim_batch(pri_3dcam_gt_to_3dwd_avg, sub_action, view_list)
            #         #prj_3dcam_to_2dim_gt = HumanCam.p3d_im2d_batch(prj_3dwd_to_3dcam_gt, sub_action, view_list)
            #         loss_consis_wd = msefun(loss_consis_weight*prj_3dwd_to_2dim_w_disr,loss_consis_weight*pos_gt_2d.permute(0,4,1,2,3)[:,[0,1,2,3],pad:pad+1].to(out.device))
            #         if summary_writer is not None and cfg.TRAIN.UNSUPERVISE==True:
            #             summary_writer.add_scalar("loss_consis_wd/iter", loss_consis_wd, iters)
            #         print('Consis Loss in world->Image is {}'.format(loss_consis_wd))

            #out = out.permute(0, 1, 4, 2,3).contiguous() #(B, T, N, J. C)
            # p3d_gt_abs = p3d_gt_abs.permute(0, 1, 4, 2, 3).contiguous()
            # print("out.shape : "+str(out.shape))
            # print("pos_gt.shape : " + str(pos_gt.shape))
            # input()
            if cfg.TRAIN.USE_INTER_LOSS:
                for i in range(len(other_out)): 
                    other_out[i] = other_out[i].permute(0, 1, 4, 2,3).contiguous() #(B, T, N, J. C)

            #print(pose_3D_in_cam_space.shape)
            #print(p3d_gt_abs[:,pad:pad+1].shape)
            loss_copy = mpjpe(pose_3D_in_cam_space , p3d_gt_abs[:,pad:pad+1])
            if summary_writer is not None and cfg.TRAIN.UNSUPERVISE==True:
                summary_writer.add_scalar("loss_copy/iter", loss_copy, iters)
            # loss = loss_copy if cfg.TRAIN.UNSUPERVISE==False else 0
            # loss +=loss_consis_wd if cfg.TRAIN.CONSIS_LOSS_ADD else 0
            # loss +=triangu_loss
            loss = loss_consis_wd
            #print('loss after add loss_consis_wd is {}'.format(loss))
            if summary_writer is not None:
                summary_writer.add_scalar("loss_final/iter", loss, iters)
            # if pred_rot is not None and cfg.TRAIN.USE_ROT_LOSS:
            #     tran_loss = msefun(pred_rot, rotation)
            #
            #     if summary_writer is not None:
            #         summary_writer.add_scalar("loss_tran/iter", tran_loss, iters)
            #     loss = loss + cfg.TRAIN.ROT_LOSS_WEIGHT * tran_loss
            
            # loss_consis_weight = cfg.TRAIN.CONSIS_LOSS_WEIGHT  if cfg.TRAIN.VISI_WEIGHT==False else vis[:,pad:pad+1,:,:,:4]
            # loss_consis = 0
            # if (cfg.H36M_DATA.PROJ_Frm_3DCAM == True) & (cfg.TRAIN.PROJ_3DCAM_TO_3DWD == False):
            #     if cfg.TRAIN.TEMPORAL_SMOOTH_LOSS_WEIGHT is not None:
            #         prj_3dpre_to_2d_full = prj_3dpre_to_2d_full.permute(0,2,3,4,1).contiguous()
            #     prj_3dpre_to_2d = prj_3dpre_to_2d.permute(0,2,3,4,1).contiguous() if cfg.TRAIN.TEMPORAL_SMOOTH_LOSS_WEIGHT is None else prj_3dpre_to_2d_full[:,pad:pad+1]
            #     loss_consis = msefun(prj_3dpre_to_2d, pos_gt_2d[:, pad:pad+1, :, :, [0,1,2,3]].to(prj_3dpre_to_2d.device)) if cfg.TRAIN.VISI_WEIGHT==False else msefun(torch.mul(prj_3dpre_to_2d, loss_consis_weight.to(prj_3dpre_to_2d.device)), torch.mul(pos_gt_2d[:, pad:pad+1, :, :, [0,1,2,3]].to(prj_3dpre_to_2d.device), loss_consis_weight.to(prj_3dpre_to_2d.device)))
            #     loss_consis += mpjpe(prj_2dpre_to_3d.to(out.device), pos_gt[:,pad:pad+1,0:4].permute(0,1,3,4,2))
            #     if cfg.TRAIN.TEMPORAL_SMOOTH_LOSS_WEIGHT is not None:
            #         loss_consis +=wd3d_pair_loss
            #         print('wd3d_pair_loss is {}'.format(wd3d_pair_loss))
            #     print('Consistancy Loss in Camera->image is {}'.format(loss_consis))
            #     if cfg.TRAIN.VISI_WEIGHT==False:
            #         loss = loss  + loss_consis_weight * loss_consis if cfg.TRAIN.CONSIS_LOSS_ADD==True else loss
            #     else:
            #         loss = loss + loss_consis if cfg.TRAIN.CONSIS_LOSS_ADD==True else loss
            #     print('Summed Loss is {}'.format(loss))
            #     summary_writer.add_scalar("loss_consis/iter", loss_consis, iters)
            
            # if cfg.TRAIN.TEMPORAL_SMOOTH_LOSS_WEIGHT is not None:
            #     smooth_err_2d = msefun(prj_3dpre_to_2d_full[:,:,1:],  prj_3dpre_to_2d_full[:,:,:-1])
            #     smooth_target = msefun(inputs_2d_pre[:,1:], inputs_2d_pre[:,:-1])
            #     err_bwt_full_mid = msefun(out, out_full.permute(0,1,4,2,3)[:,pad:pad+1])
            #     #weight_smooth = cfg.TRAIN.TEMPORAL_SMOOTH_LOSS_WEIGHT * 1/len(cfg.H36M_DATA.TRAIN_CAMERAS)
            #     #smooth_err_3d = msefun(out_full[:,1:], out_full.permute(0,1,4,2,3)[:,:-1])
            #     smooth_err = (smooth_err_2d - smooth_target) if cfg.TRAIN.ERR_BTW_FULL_MID==False else (smooth_err_2d - smooth_target) + err_bwt_full_mid
            #     print('Smooth_err is {}'.format(smooth_err))
            #     summary_writer.add_scalar("smooth_err/iter", smooth_err, iters)
            #     loss += smooth_err if cfg.TRAIN.SMOOTH_LOSS_ADD else 0

            inter_loss_weight = cfg.TRAIN.INTER_LOSS_WEIGHT
            inter_loss_all = 0
            if cfg.TRAIN.USE_INTER_LOSS:
                for i in range(len(other_out)):
                    if other_out[i].shape[1] == 1:
                        inter_loss = mpjpe(other_out[i] , pos_gt[:,pad:pad+1]) if cfg.TRAIN.UNSUPERVISE==False else 0
                    else:
                        inter_loss = mpjpe(other_out[i] , pos_gt)if cfg.TRAIN.UNSUPERVISE==False else 0
                    inter_loss_all = inter_loss_all + inter_loss_weight[i] * inter_loss 
                    if summary_writer is not None:
                        summary_writer.add_scalar("loss_inter_{}/iter".format(cfg.TRAIN.INTER_LOSS_NAME[i]), inter_loss, iters)
            
            mv_loss_all = 0
            if cfg.TRAIN.USE_MV_LOSS and epoch >= 0:
                mv_loss = mv_mpjpe(other_out[-1], pos_gt[:,pad:pad+1], mask) if other_out[-1] is not None else mv_mpjpe(other_out[0], pos_gt[:,pad:pad+1], mask)
                mv_loss_all = mv_loss_all + cfg.TRAIN.MV_LOSS_WEIGHT * mv_loss
                if summary_writer is not None:
                    summary_writer.add_scalar("loss_mv_loss/iter", mv_loss, iters)

            loss_total = loss
            print('Unsupervised Loss is {}'.format(loss_total))
            print('Supervised Loss is {}'.format(loss_copy))
            if cfg.TRAIN.USE_INTER_LOSS:
                loss_total = loss_total + inter_loss_all 
            if cfg.TRAIN.USE_MV_LOSS and epoch >= 0:
                loss_total = loss_total + mv_loss_all 
            #print('Loss_total is {}'.format(loss_total))
            loss_total.backward()

            optimizer.step()
            iters += 1

        process.close() 

        ###########eval
        with torch.no_grad():
            if not cfg.TEST.TRIANGULATE:
                load_state(model, model_test)
                model_test.eval()
            NUM_VIEW = len(cfg.H36M_DATA.TEST_CAMERAS)
            if EVAL:
                eval_start_time = time.time()
            for t_len in cfg.TEST.NUM_FRAMES:
                epoch_loss_valid = 0  
                action_mpjpe = {}
                for act in actions:
                    action_mpjpe[act] = [0] * NUM_VIEW
                    for i in range(NUM_VIEW):
                        action_mpjpe[act][i] = [0] * (NUM_VIEW + 1)
                N = [0] * NUM_VIEW
                for i in range(NUM_VIEW):
                    N[i] = [0] * (NUM_VIEW + 1)
                    
                for num_view in cfg.TEST.NUM_VIEWS:
                    pad_t = t_len // 2
                    for view_list in itertools.combinations(list(range(NUM_VIEW)), num_view):
                        view_list = list(view_list)
                        N[num_view - 1][-1] += 1
                        for i in view_list:
                            N[num_view - 1][i] += 1
                        for valid_subject in cfg.H36M_DATA.SUBJECTS_TEST:
                            for act in vis_actions if EVAL else actions:
                                poses_valid_2d = fetch([valid_subject], [act], is_test =True)

                                test_generator = ChunkedGenerator(cfg.TEST.BATCH_SIZE, poses_valid_2d, 1,pad=pad_t, causal_shift=causal_shift, shuffle=False, augment=False,kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right, extra_poses_3d=poses_3d_extra)
                                for batch_2d, _, _ in test_generator.next_epoch():
                                    inputs = torch.from_numpy(batch_2d.astype('float32'))
                                    inputs_2d_gt = inputs[...,:2,:]
                                    inputs_2d_pre = inputs[...,2:4,:]

                                    cam_3d = inputs[..., 4:7,:]
                                    vis = inputs[...,7:8,:]
                                    inputs_3d_gt = cam_3d[:,pad_t:pad_t+1]
                                    p3d_root = copy.deepcopy(inputs_3d_gt[:, :, :1])  # (B,T, 1, 3, N)
                                    inputs_3d_gt[:,:,0] = 0
                                    #p3d_gt_abs = pos_gt + p3d_root
                                    inputs_3d_gt = inputs_3d_gt + p3d_root
                                    if use_2d_gt:
                                        inp = inputs_2d_gt
                                        vis = torch.ones(*vis.shape)
                                    else:
                                        inp = inputs_2d_pre

                                    inp = inp[...,view_list] #B, T,V, C, N
                                    inp = torch.cat((inp, vis[..., view_list]), dim = -2)
                                    B = inp.shape[0]
                                    # if cfg.TEST.TRIANGULATE:
                                    #     trj_3d = HumanCam.p2d_cam3d(inp[:, pad_t:pad_t+1, :,:2, :], valid_subject, view_list)#B, T, J, 3, N)
                                    #     loss = 0
                                    #     for idx, view_idx in enumerate(view_list):
                                    #         loss_view_tmp = eval_metrc(cfg, trj_3d[..., idx], inputs_3d_gt[..., view_idx])
                                    #         loss += loss_view_tmp.item()
                                    #         action_mpjpe[act][num_view - 1][view_idx] += loss_view_tmp.item() * inputs_3d_gt.shape[0]
                                    #
                                    #     action_mpjpe[act][num_view - 1][-1] += loss * inputs_3d_gt.shape[0]
                                    #     continue

                                    # inp_flip = copy.deepcopy(inp)
                                    # inp_flip[:,:,:,0] *= -1
                                    # inp_flip[:,:,joints_left + joints_right] = inp_flip[:,:,joints_right + joints_left]
                                    # if cfg.NETWORK.USE_GT_TRANSFORM:
                                    #     inputs_3d_gt_flip = copy.deepcopy(inputs_3d_gt)
                                    #     inputs_3d_gt_flip[:,:,:,0] *= -1
                                    #     inputs_3d_gt_flip[:,:,joints_left + joints_right] = inputs_3d_gt_flip[:,:,joints_right + joints_left]
                                    # if cfg.TEST.TEST_FLIP:
                                    #     if cfg.NETWORK.USE_GT_TRANSFORM:
                                    #         rotation = get_rotation(torch.cat((inputs_3d_gt, inputs_3d_gt_flip), dim = 0)[..., view_list])
                                    #         rotation = rotation.repeat(1, 1, 1, inp.shape[1], 1, 1)
                                    #     else:
                                    #         rotation = None
                                    #     print('Evaluation part! Input size={}, input_fip={}'.format(inp.shape, inp_flip.shape))
                                    #     out, other_info = model_test(torch.cat((inp, inp_flip), dim = 0), rotation)
                                    #     r_out = out
                                    #
                                    #     out[B:,:,:,0] *= -1
                                    #     out[B:,:,joints_left + joints_right] = out[B:,:,joints_right + joints_left]
                                    #
                                    #     out = (out[:B] + out[B:]) / 2
                                    # else:
                                    #     if cfg.NETWORK.USE_GT_TRANSFORM:
                                    #         rotation = get_rotation(inputs_3d_gt[..., view_list])
                                    #         rotation = rotation.repeat(1, 1, 1, inp.shape[1], 1, 1)
                                    #     else:
                                    rotation = None#get_rotation(inputs_3d_gt[..., view_list])
                                    #rotation = rotation.repeat(1, 1, 1, inp.shape[1], 1, 1)
                                    out, other_info = model_test(inp, rotation)
                                    #out[:,:,0] = 0
                                    
                                    out = out.detach().cpu()
                                    out = torch.mean(out, -1)
                                    sub_action = [[valid_subject, act] for k in range(inputs.shape[0])]
                                    pose_3D_in_cam_space = HumanCam.p3dwd_p3dcam_batch(out[:, 0, :, :], sub_action, view_list)
                                    pose_3D_in_cam_space = pose_3D_in_cam_space.permute(0, 2, 3, 1)
                                    out = torch.unsqueeze(pose_3D_in_cam_space, 1)

                                    """for v in range(4):
                                        x_out = out[0, 0, :, 0, v]
                                        y_out = out[0, 0, :, 1, v]
                                        z_out = out[0, 0, :, 2, v]

                                        x_gt = inputs_3d_gt[0, 0, :, 0, v]
                                        y_gt = inputs_3d_gt[0, 0, :, 1, v]
                                        z_gt = inputs_3d_gt[0, 0, :, 2, v]

                                        fig = plt.figure("view "+str(v))
                                        ax = fig.add_subplot(projection='3d')
                                        ax.scatter(x_gt, y_gt, z_gt, marker='o', color='r', label="gt 3D")
                                        ax.scatter(x_out, y_out, z_out, marker='+', color='g', label="pred 3D")
                                        ax.set_box_aspect((np.ptp(x_gt), np.ptp(y_gt), np.ptp(z_gt)))
                                    print(x_out - x_gt)
                                    print(y_out - y_gt)
                                    print(z_out - z_gt)
                                    plt.show()"""

                                    # if EVAL and args.vis_3d:
                                    #     vis_tool.show(inputs_2d_pre[:,pad_t], out[:,0], inputs_3d_gt[:,0])
                                    #
                                    # if cfg.TEST.TEST_ROTATION:
                                    #     out = test_multi_view_aug(out, vis[...,view_list])
                                    #     out[:,:,0] = 0
                                    
                                    # if cfg.NETWORK.USE_GT_TRANSFORM and EVAL and len(view_list) > 1 and cfg.TEST.ALIGN_TRJ:
                                    #     #TODO: 使用T帧姿态进行三角剖分得到平均骨骼长度再对齐
                                    #     trj_3d = HumanCam.p2d_cam3d(inp[:, pad_t:pad_t+1, :,:2, :], valid_subject, view_list)#B, T, J, 3, N)
                                    #     out_align = align_target_numpy(cfg, out, trj_3d)
                                    #     out_align[:,:,0] = 0
                                    #     out = out_align

                                    loss = 0 
                                    for idx, view_idx in enumerate(view_list):
                                        loss_view_tmp = eval_metrc(cfg, out[..., idx], inputs_3d_gt[..., view_idx])
                                        loss += loss_view_tmp.item()
                                        action_mpjpe[act][num_view - 1][view_idx] += loss_view_tmp.item() * inputs_3d_gt.shape[0]

                                    action_mpjpe[act][num_view - 1][-1] += loss * inputs_3d_gt.shape[0]
                print('num_actions :{}'.format(len(action_frames)))
                for num_view in cfg.TEST.NUM_VIEWS:
                    tmp = [0] * (NUM_VIEW + 1)
                    print('num_view:{}'.format(num_view))
                    for act in action_mpjpe:
                        for i in range(NUM_VIEW):
                            action_mpjpe[act][num_view - 1][i] /= (action_frames[act] * N[num_view - 1][i])
                        action_mpjpe[act][num_view - 1][-1] /= (action_frames[act] * N[num_view - 1][-1] * num_view)
                        print('mpjpe of {:18}'.format(act), end = ' ')
                        for i in range(NUM_VIEW):
                            print('view_{}: {:.3f}'.format(cfg.H36M_DATA.TEST_CAMERAS[i], action_mpjpe[act][num_view - 1][i] * 1000), end = '    ')
                            tmp[i] += action_mpjpe[act][num_view - 1][i] * 1000
                            
                        print('avg_action: {:.3f}'.format(action_mpjpe[act][num_view - 1][-1] * 1000))
                        tmp[-1] += action_mpjpe[act][num_view - 1][-1] * 1000
                    print('avg:', end = '                        ')
                    for i in range(NUM_VIEW):
                        print('view_{}: {:.3f}'.format(i, tmp[i] / len(action_frames)), end = '    ')
                    print('avg_all   : {:.3f}'.format(tmp[-1] / len(action_frames)))
                        
                    if summary_writer is not None:
                        summary_writer.add_scalar("test_mpjpe_t{}_v{}/epoch".format(t_len, num_view), tmp[-1] / len(action_frames), epoch)
                    epoch_loss_valid += tmp[-1] / len(action_frames)
                epoch_loss_valid /= len(cfg.TEST.NUM_VIEWS)
                print('t_len:{} avg:{:.3f}'.format(t_len, epoch_loss_valid))
                if summary_writer is not None:
                        summary_writer.add_scalar("epoch_loss_valid/epoch", epoch_loss_valid, epoch)
                
            if EVAL:
                eval_elapsed = (time.time() - eval_start_time)/60
                print('time:{:.2f}'.format(eval_elapsed))
                exit()
            
            
        epoch += 1
        
        if epoch_loss_valid < best_result:
            best_result = epoch_loss_valid
            best_state_dict = copy.deepcopy(model.module.state_dict())
            best_result_epoch = epoch
        elapsed = (time.time() - start_time)/60
        print('epoch:{:3} time:{:.2f} lr:{:.9f} best_result_epoch:{:3} best_result:{:.3f}'.format(epoch, elapsed, lr, best_result_epoch, best_result))
        print('checkpoint:{}'.format(cfg.TRAIN.CHECKPOINT))
        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        momentum = initial_momentum * np.exp(-epoch/cfg.TRAIN.NUM_EPOCHES * np.log(initial_momentum/final_momentum))
        model.module.set_bn_momentum(momentum)
            
        torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model':model.module.state_dict(),
                'best_epoch': best_result_epoch,
                'best_model':best_state_dict,
            }, cfg.TRAIN.CHECKPOINT)
          

