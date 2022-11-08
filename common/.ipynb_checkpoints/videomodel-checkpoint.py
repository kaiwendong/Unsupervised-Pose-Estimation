import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import sys
import copy
#import torch.cuda.amp as amp
BN_MOMENTUM = 0.1
head_joint_idx = [0, 1, 4, 7, 8, 11, 14, 9, 10]
hand_joint_idx = [0, 1, 4, 7, 8, 11, 14, 12, 13, 15, 16]
foot_joint_idx = [0, 1, 4, 7, 8, 11, 14,  2, 3, 5, 6, ]
hand_joint_left_idx = [0, 1, 4, 7, 8, 11, 14, 12, 13]
hand_joint_right_idx = [0, 1, 4, 7, 8, 11, 14, 15, 16]
foot_joint_right_idx = [0, 1, 4, 7, 8, 11, 14,  2, 3, ]
foot_joint_left_idx = [0, 1, 4, 7, 8, 11, 14,  5, 6, ]
common_joint_idx = [0, 1, 4, 7, 8, 11, 14]
head_bone_idx = [0, 3, 7, 10, 6]
hand_bone_idx = [0, 3, 7, 10, 8, 9, 11, 12, 17, 18, 19, 20]
hand_bone_left_idx  = [0, 3, 7, 10, 8, 9,  17, 18, ]
hand_bone_right_idx  = [0, 3, 7, 10,11, 12,  19, 20]

foot_bone_idx = [0, 3, 7, 10, 1, 2, 4, 5, 13, 14, 15, 16]
foot_bone_left_idx = [0, 3, 7, 10,  4, 5, 15, 16]
foot_bone_right_idx = [0, 3, 7, 10, 1, 2, 13, 14,]
def single_to_video(pre, N = 50, K = 7):
    NF = N
    
    out = copy.deepcopy(pre)#(F, C1, C2)
    B, C1, C2 = out.shape
    try:
        for i in range(int(out.shape[0] / NF + 1)):
            st = i*NF
            end = (i+1)*NF if out.shape[0] > (i+1) * NF else out.shape[0]
            if (st == end or st == out.shape[0]):
                break
            F = NF if out.shape[0] > (i+1) * NF else out.shape[0] - i*NF
            x = np.arange(F)
            fixed_bases = [np.ones([F]) * np.sqrt(0.5)]
            for i in range(1, K):
                fixed_bases.append(np.cos(i * np.pi * ((x + 0.5) / F)))
            fixed_bases = np.array(fixed_bases)
            bases_tmp = torch.from_numpy(fixed_bases).float().cuda()
            out_tmp = copy.deepcopy(out[st:end].view(F, -1))#(F, C1, C2)

            out_tmp = out_tmp.permute(1, 0)

            out_tmp = torch.matmul(out_tmp, torch.transpose(bases_tmp, 0, 1)) / F * 2

            out_tmp = torch.matmul(out_tmp, bases_tmp)

            out_tmp = out_tmp.view(C1, C2, F)
            out_tmp = out_tmp.permute(2, 0, 1)
            out[st:end] = out_tmp
            if out.shape[0] <= (i + 1) * NF:
                break
    except:
        print(st, end, out.shape[0])
        sys.exit()
    return out

import torch.nn as nn

    

class TemporalModel(nn.Module):
    def __init__(self, in_channels, filter_widths, dropout=0.25, channels=1024,):

        super().__init__()
        
        self.expand_conv = nn.Conv1d(in_channels, channels, filter_widths[0], bias=False)
        self.expand_bn = nn.BatchNorm1d(channels, momentum = 0.1)
        self.pad = [ filter_widths[0] // 2 ]
        layers_conv = []
        layers_bn = []
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.causal_shift = [0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append( 0)
            
            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] ,
                                         dilation=next_dilation,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
    def forward(self, pos_2d, bone_angle):
        B,T, V1, C1 = pos_2d.shape
        B,T, V2, C2 = bone_angle.shape
        
        pos_2d = pos_2d.view(B, T, V1 * C1).contiguous()
        bone_angle = bone_angle.view(B,T, V2 * C2,).contiguous()
        x = torch.cat((pos_2d, bone_angle), dim = -1)
        x = x.permute(0, 2, 1).contiguous()
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        return x
    
class TemporalModelOptimized1f(nn.Module):
    def __init__(self,in_channels,filter_widths, dropout=0.25, channels=1024):
        super().__init__()
        
        self.expand_conv = nn.Conv1d(in_channels, channels, filter_widths[0], stride=filter_widths[0], bias=False)
        self.expand_bn = nn.BatchNorm1d(channels, momentum = 0.1)
        self.pad = [ filter_widths[0] // 2 ]
        layers_conv = []
        layers_bn = []
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.causal_shift = [ 0 ]
        self.filter_widths = filter_widths
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append( 0)
            
            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
    def forward(self, pos_2d, bone_angle):
        B,T, V1, C1 = pos_2d.shape
        B,T, V2, C2 = bone_angle.shape
        
        pos_2d = pos_2d.view(B, T, V1 * C1).contiguous()
        bone_angle = bone_angle.view(B,T, V2 * C2,).contiguous()
        x = torch.cat((pos_2d, bone_angle), dim = -1)
        x = x.permute(0, 2, 1).contiguous()
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        

        return x


class MyVideoTrainModel(nn.Module):
    def __init__(self,filter_widths,  dropout=0.25, channels=1024):
        super().__init__()
        in_channels = len(head_joint_idx) * 2 + len(head_bone_idx) * 2
        
        self.head_model = TemporalModelOptimized1f(in_channels = in_channels, channels = channels,dropout = dropout, filter_widths = filter_widths)
        in_channels = len(hand_joint_left_idx) * 2 + len(hand_bone_left_idx) * 2
        self.hand_left_model = TemporalModelOptimized1f(in_channels = in_channels, channels = channels,dropout = dropout, filter_widths = filter_widths )
        in_channels = len(foot_joint_left_idx) * 2 + len(foot_bone_left_idx) * 2
        self.foot_left_model = TemporalModelOptimized1f(in_channels = in_channels, channels = channels,dropout = dropout, filter_widths = filter_widths)
        
        in_channels = len(hand_joint_right_idx) * 2 + len(hand_bone_right_idx) * 2
        self.hand_right_model = TemporalModelOptimized1f(in_channels = in_channels, channels = channels,dropout = dropout, filter_widths = filter_widths)
        in_channels = len(foot_joint_right_idx) * 2 + len(foot_bone_right_idx) * 2
        self.foot_right_model = TemporalModelOptimized1f(in_channels = in_channels, channels = channels,dropout = dropout, filter_widths = filter_widths)
        self.shrink_conv =nn.Conv1d(channels * 5, channels, 1, bias = False)
        self.shrink_bn = nn.BatchNorm1d(channels, momentum = 0.1)
        self.num_layers = 0
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv1d(channels,channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channels, momentum = 0.1))
            conv_layers.append(nn.Conv1d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channels, momentum = 0.1))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
                             
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        
        self.shrink = nn.Conv1d(channels, 17 * 3, 1)
    def set_bn_momentum(self, momentum):
        self.head_model.set_bn_momentum(momentum)
        self.hand_left_model.set_bn_momentum(momentum)
        self.foot_left_model.set_bn_momentum(momentum)
        self.hand_right_model.set_bn_momentum(momentum)
        self.foot_right_model.set_bn_momentum(momentum)
        self.shrink_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, pos_2d, bone_angle):
        head = self.head_model(pos_2d[:,:,head_joint_idx], bone_angle[:,:,head_bone_idx])
        hand_left = self.hand_left_model(pos_2d[:,:,hand_joint_left_idx], bone_angle[:,:, hand_bone_left_idx])
        foot_left = self.foot_left_model(pos_2d[:,:,foot_joint_left_idx], bone_angle[:, :,foot_bone_left_idx])
        hand_right = self.hand_right_model(pos_2d[:,:,hand_joint_right_idx], bone_angle[:,:, hand_bone_right_idx])
        foot_right = self.foot_right_model(pos_2d[:,:,foot_joint_right_idx], bone_angle[:,:, foot_bone_right_idx])
        f =torch.cat((head, hand_left,hand_right, foot_left, foot_right), dim = 1)
        f = self.drop(self.relu(self.shrink_bn(self.shrink_conv(f))))
        for i in range(self.num_layers):
            res = f
            f = self.drop(self.relu(self.bn_layers[i * 2](self.conv_layers[i * 2](f))))
            f = self.drop(self.relu(self.bn_layers[i * 2 + 1](self.conv_layers[i * 2 + 1](f))))
            f = f + res

        out = self.shrink(f)
        out = out.view(-1, 17, 3)
        return out
class MyVideoTestModel(nn.Module):
    def __init__(self,filter_widths,  dropout=0.25, channels=1024):
        super().__init__()
        in_channels = len(head_joint_idx) * 2 + len(head_bone_idx) * 2
        
        self.head_model = TemporalModel(in_channels = in_channels, channels = channels,dropout = dropout, filter_widths = filter_widths)
        in_channels = len(hand_joint_left_idx) * 2 + len(hand_bone_left_idx) * 2
        self.hand_left_model = TemporalModel(in_channels = in_channels, channels = channels,dropout = dropout, filter_widths = filter_widths )
        in_channels = len(foot_joint_left_idx) * 2 + len(foot_bone_left_idx) * 2
        self.foot_left_model = TemporalModel(in_channels = in_channels, channels = channels,dropout = dropout, filter_widths = filter_widths)
        
        in_channels = len(hand_joint_right_idx) * 2 + len(hand_bone_right_idx) * 2
        self.hand_right_model = TemporalModel(in_channels = in_channels, channels = channels,dropout = dropout, filter_widths = filter_widths)
        in_channels = len(foot_joint_right_idx) * 2 + len(foot_bone_right_idx) * 2
        self.foot_right_model = TemporalModel(in_channels = in_channels, channels = channels,dropout = dropout, filter_widths = filter_widths)
        self.shrink_conv =nn.Conv1d(channels * 5, channels, 1, bias = False)
        self.shrink_bn = nn.BatchNorm1d(channels, momentum = 0.1)
        self.num_layers = 0
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv1d(channels,channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channels, momentum = momentum))
            conv_layers.append(nn.Conv1d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channels, momentum = momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
                             
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        
        self.shrink = nn.Conv1d(channels, 17 * 3, 1)
    def set_bn_momentum(self, momentum):
        self.head_model.set_bn_momentum(momentum)
        self.hand_left_model.set_bn_momentum(momentum)
        self.foot_left_model.set_bn_momentum(momentum)
        self.hand_right_model.set_bn_momentum(momentum)
        self.foot_right_model.set_bn_momentum(momentum)
        self.shrink_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, pos_2d, bone_angle):
        head = self.head_model(pos_2d[:,:,head_joint_idx], bone_angle[:,:,head_bone_idx])
        hand_left = self.hand_left_model(pos_2d[:,:,hand_joint_left_idx], bone_angle[:, :,hand_bone_left_idx])
        foot_left = self.foot_left_model(pos_2d[:,:,foot_joint_left_idx], bone_angle[:,:, foot_bone_left_idx])
        hand_right = self.hand_right_model(pos_2d[:,:,hand_joint_right_idx], bone_angle[:,:, hand_bone_right_idx])
        foot_right = self.foot_right_model(pos_2d[:,:,foot_joint_right_idx], bone_angle[:, :,foot_bone_right_idx])
        f =torch.cat((head, hand_left,hand_right, foot_left, foot_right), dim = 1)
        f = self.drop(self.relu(self.shrink_bn(self.shrink_conv(f))))
        for i in range(self.num_layers):
            res = f
            f = self.drop(self.relu(self.bn_layers[i * 2](self.conv_layers[i * 2](f))))
            f = self.drop(self.relu(self.bn_layers[i * 2 + 1](self.conv_layers[i * 2 + 1](f))))
            f = f + res
        
        out = self.shrink(f)
        B,C, T = out.shape
        out = out.view(-1, 17, 3, T)
        out = out.permute(0, 3, 1, 2)
        return out