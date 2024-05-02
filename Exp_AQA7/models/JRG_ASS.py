# coding=utf-8
# We thank Jia-Hui Pan for offering the preprocessed datasets and source code.
# The model is modified from the official implementation of JRG-ASS model proposed by [1] and [2]. 
#     [1] Pan, Jia-Hui, Jibin Gao, and Wei-Shi Zheng. "Action assessment by joint relation graphs." Proceedings of the IEEE/CVF international conference on computer vision. 2019.
#     [2] Pan, Jia-Hui, Jibin Gao, and Wei-Shi Zheng. "Adaptive action assessment." IEEE Transactions on Pattern Analysis and Machine Intelligence 44.12 (2021): 8779-8795.

from builtins import print
import os
import numpy as np 
import copy
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.autograd import *
from einops import rearrange, repeat, reduce

def get_loss(pred, labels, type='new_mse', action_id=0):
    bias_ = 1e-6
    if type == 'new_mse':
        mean_pred = torch.mean(pred)
        mean_labels = torch.mean(labels)
        normalized_pred = (pred - mean_pred) / torch.sqrt(torch.var(pred)+bias_)
        normalized_labe = (labels - mean_labels) / torch.sqrt(torch.var(labels)+bias_)
        loss_new_mse = torch.mean((normalized_pred - normalized_labe)**2, dim=0)
        return loss_new_mse * 100.0
    elif type == 'pearson':
        mean_pred = torch.mean(pred)
        mean_labels = torch.mean(labels)
        loss_pearson =  torch.tensor(1.0).cuda() - torch.sum((pred - mean_pred) * (labels - mean_labels)) \
            / (torch.sqrt(torch.sum((pred - mean_pred)**2) * torch.sum((labels - mean_labels)**2) + bias_))
        return loss_pearson * 100.0
    elif type == 'mse':
        loss_mse = torch.mean((pred - labels)**2, dim=0)
        return loss_mse 
    elif type == 'huber':
        crit = torch.nn.SmoothL1Loss()
        return crit(pred, labels)
    else:
        return None

def build_joint_graphs(joint_num, hop_num=4):
    a_file = './mat_a.npy'
    if not os.path.isfile(a_file):
        a_file = '/home/yuanming/Code/IAQA/ablation/mat_a.npy'
    if joint_num == 17:
        graph = np.load(a_file).astype(float) + np.identity(joint_num)
    else:
        graph = np.array([[1, 1, 1, 0],
                          [1, 1, 0, 1],
                          [1, 0, 1, 0],
                          [0, 1, 0, 1]])
    graph = torch.from_numpy(graph).int()
    graphs = [graph, ]
    for i in range(hop_num-1):
        tmp = torch.matmul(graphs[-1], graph)
        tmp[tmp != 0] = 1
        graphs.append(tmp)
    for i in range(hop_num-1, 0, -1):
        graphs[i] = graphs[i] * (1-graphs[i-1])
    return graphs


def build_spatial_temporal_mat(joint_num, hop_num=4, save_graph=False, task_num=6):
    if not save_graph:
        mats = nn.ModuleList([
            nn.Linear(joint_num, joint_num, bias=False)
            for _ in range(hop_num)
        ])
    else:
        mats = nn.ModuleList([
            nn.ModuleList([nn.Linear(joint_num, joint_num, bias=False) for x in range(task_num)])
            for _ in range(hop_num)
        ])
    return mats


class ASS_JRG(nn.Module):
    def __init__(self, whole_size=400, patch_size=400, seg_num=12, joint_num=17, out_dim=1, mode='', save_graph=False,
             feature_id_to_remove=[], task_list=[], G_E_graph=False, alpha=0.5):
        super(ASS_JRG, self).__init__()
        self.hop_num = 4
        self.task_num = 6

        self.whole_size = whole_size
        self.patch_size = patch_size
        self.seg_num = seg_num
        self.joint_num = joint_num
        self.module_num = 12
        self.module_num -= len(feature_id_to_remove)
        self.mode = mode
        self.task_list = task_list
        self.dropout_rate = 0.1  # 0.2
        self.hidden1 = 256
        self.hidden2 = 256
        self.hidden3 = 32
        self.alpha = alpha
        self.save_graph = save_graph
        self.fix_graph = True
        self.out_dim = out_dim

        self.register_buffer("joint_graphs", torch.stack(build_joint_graphs(joint_num), dim=0))

        self.register_parameter("spatial_mats",
                                nn.Parameter(torch.zeros(self.task_num, self.hop_num, joint_num, joint_num)))
        self.register_parameter("temporal_mats",
                                nn.Parameter(torch.zeros(self.task_num, self.hop_num, joint_num, joint_num)))

        self.g_e_graph = G_E_graph

        self.register_parameter("general_spatial_mats",
                                nn.Parameter(torch.randn(self.hop_num, joint_num, joint_num)))
        self.register_parameter("general_temporal_mats",
                                nn.Parameter(torch.randn(self.hop_num, joint_num, joint_num)))
        # Aggregators from Joint Difference Module
        self.register_parameter("spatial_JCWs", nn.Parameter(torch.randn(self.hop_num, joint_num, 1)))
        self.register_parameter("temporal_JCWs", nn.Parameter(torch.randn(self.hop_num, joint_num, 1)))
        
        # Feature Encoders
        self.encoders_whole = nn.ModuleList([self.build_encoder(self.whole_size) for _ in range(1*2)]) # (rgb, flow)
        self.encoders_diffwhole = nn.ModuleList([self.build_encoder(self.whole_size) for _ in range(1*2)]) # (rgb, flow)
        self.encoders_comm0 = nn.ModuleList([self.build_encoder(self.patch_size) for _ in range(1*2)])
        self.encoders_comm1 = nn.ModuleList([self.build_encoder(self.patch_size) for _ in range(self.hop_num*2)])       # these parameters are not used for training
        self.encoders_diff0 = nn.ModuleList([self.build_encoder(self.patch_size) for _ in range(self.hop_num*2)])
        self.encoders_diff1 = nn.ModuleList([self.build_encoder(self.patch_size) for _ in range(self.hop_num*2)])

        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(True)
        )
        self.last_fuse = nn.Linear(self.module_num, 1, bias=False)       # these parameters are not used for training

    def forward(self, feat_whole, feat_patch):
        """
            feat_whole: whole-scene feature
            feat_patch: joint feature
            action_class: action class id
        """
        B, J, T, D = feat_patch.shape
        if self.save_graph:
            spatial_graphs = torch.abs(self.spatial_mats * self.joint_graphs)  # 6，4，17，17
            temporal_graphs = torch.abs(self.temporal_mats * self.joint_graphs)
        
        if self.g_e_graph:
            general_spatial_graphs = torch.abs(self.general_spatial_mats * self.joint_graphs)
            general_temporal_graphs = torch.abs(self.general_temporal_mats * self.joint_graphs)
            spatial_graphs = (1-self.alpha) * spatial_graphs + self.alpha * general_spatial_graphs
            temporal_graphs = (1-self.alpha) * temporal_graphs + self.alpha * general_temporal_graphs   # 6，4，17，17
        
        elif not self.save_graph:
            general_spatial_graphs = torch.abs(self.general_spatial_mats * self.joint_graphs)
            general_temporal_graphs = torch.abs(self.general_temporal_mats * self.joint_graphs)
            spatial_graphs = general_spatial_graphs.unsqueeze(0)
            temporal_graphs = general_temporal_graphs.unsqueeze(0)           

        bs = feat_whole.shape[0]
        
        comm_H0 = rearrange(feat_patch, 'B J T D -> B T J D')
        comm_h1s = rearrange(torch.matmul(rearrange(feat_patch, 'B J T D -> (B T D) J'), spatial_graphs),
                             'A H (B T D) J ->A H B T J D', B=B, D=D)

        diff_mat_fp0 = rearrange(feat_patch, 'B J T D -> B T D J')
        diff_mat_fp1 = torch.cat([diff_mat_fp0[:, 1:], diff_mat_fp0[:, -1].unsqueeze(dim=1)], dim=1)
        diff_mat_f0 = diff_mat_fp0[..., None, :] - diff_mat_fp0[..., None]      # B T D J J
        diff_mat_f1 = diff_mat_fp1[..., None, :] - diff_mat_fp0[..., None]


        diff_d0 = torch.matmul(diff_mat_f0.reshape(-1, J, J) * spatial_graphs.unsqueeze(dim=2),
                         self.spatial_JCWs.unsqueeze(dim=1))
        diff_d1 = torch.matmul(diff_mat_f1.reshape(-1, J, J) * temporal_graphs.unsqueeze(dim=2),
                        self.temporal_JCWs.unsqueeze(dim=1))
        diff_d0 = rearrange(diff_d0, 'A H (B T D) J 1 -> A H B T J D', B=B, T=T, D=D)
        diff_d1 = rearrange(diff_d1, 'A H (B T D) J 1 -> A H B T J D', B=B, T=T, D=D)
        

        diff_mat_fp1 = torch.cat([feat_whole[:, 1:feat_whole.shape[1]], feat_whole[:, -1].unsqueeze(dim=1)], dim=1)
        feat_diff = torch.abs(diff_mat_fp1 - feat_whole)
        
        # Encoding shape(batch, seg_num, joint_num, hidden1)
        rgb_t = torch.ones(bs).cuda()
        rgb_f = torch.zeros(bs).cuda()
        rgb_feat, rgb_whole = self.encode_feats(feat_whole, feat_diff, comm_H0, comm_h1s, diff_d0, diff_d1, rgb=rgb_t)
        flow_feat, flow_whole = self.encode_feats(feat_whole, feat_diff, comm_H0, comm_h1s, diff_d0, diff_d1, rgb=rgb_f)
        
        cell = torch.cat([rgb_feat, flow_feat], dim=-1)
       
        fused_feat = torch.cat([rgb_feat+flow_feat, rgb_feat+flow_feat], dim=5)
        fused_feat = reduce(fused_feat, 'A B T J X D -> A B D', 'mean')
        fused_feat = self.regressor(fused_feat).transpose(0,1)
        return fused_feat, None

    def encode_feats(self, feat_whole, feat_diff, comm_H0, comm_h1s, diff_d0, diff_d1, rgb=True):
        assert self.patch_size == self.whole_size
        idx, begin, end = 0, 0, self.patch_size//2
        if rgb[0]==0:
            idx, begin, end = 1, self.patch_size//2, self.patch_size
        _encoded_whole = self.encoders_whole[idx](feat_whole[..., begin:end])    # 0
        encoded_diff = self.encoders_diffwhole[idx](feat_diff[..., begin:end]) # 0
        encoded_comm0 = self.encoders_comm0[idx](comm_H0[..., begin:end]) # 1
        encoded_comm0 = repeat(encoded_comm0, 'B T J D -> A B T J D', J=self.joint_num, A=comm_h1s.shape[0])

        encoded_comm1 = [self.encoders_comm0[idx](x[..., begin:end]) for x in comm_h1s.transpose(0,1)]
        encoded_diff0 = [self.encoders_diff0[idx](x[..., begin:end]) for x in diff_d0.transpose(0,1)]
        encoded_diff1 = [self.encoders_diff1[idx](x[..., begin:end]) for x in diff_d1.transpose(0,1)]
        encoded_whole = repeat(_encoded_whole, 'B T D -> A B T J D', J=self.joint_num, A=comm_h1s.shape[0])
        encoded_diff = repeat(encoded_diff, 'B T D -> A B T J D', J=self.joint_num, A=comm_h1s.shape[0])
        all_feats = [encoded_whole, encoded_diff, encoded_comm0] + encoded_comm1 + encoded_diff0 + encoded_diff1
        out = torch.stack(all_feats, dim=3)
        return out, _encoded_whole

    def build_encoder(self, input_size):
        return nn.Sequential(
            nn.Linear(input_size//2, self.hidden1//2),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True)
        )


		
def get_numpy_mse(pred, score):
	pred = np.array(pred)
	score = np.array(score)
	return np.sum((pred - score)**2) / pred.shape[0]

def get_numpy_spearman(pred, score):
	pred = np.array(pred)
	score = np.array(score)
	import pdb
	#pdb.set_trace()
	return stats.spearmanr(pred, score).correlation

def get_numpy_pearson(pred, score):
	pred = np.array(pred)
	score = np.array(score)
	return stats.pearsonr(pred, score)[0]

def run_jrg(model_, feat_whole, feat_patch, save_graph=False, seen_tasks=[], is_train=False, args=None):
    batch_size = feat_whole.shape[0]
    fused_feat = None
    fused_feat, featmap_list = model_(feat_whole, feat_patch)       # [1,B, 256] [B, 12, 17, 15, 256] 
    fused_feat = fused_feat.transpose(0,1)
    if not save_graph:
        fused_feat = fused_feat[0]
    return fused_feat, featmap_list

def init_e_graph(model_, t, seen_tasks=[]):
    if t==0:
        alpha = model_.module.alpha
        g_spatial_mat = copy.deepcopy(model_.module.general_spatial_mats)
        g_temporal_mat = copy.deepcopy(model_.module.general_temporal_mats)
        model_.module.spatial_mats[seen_tasks[0]] = (1-alpha) * g_spatial_mat
        model_.module.temporal_mats[seen_tasks[0]] = (1-alpha) * g_temporal_mat
    else:
        model_.module.spatial_mats[seen_tasks[-1]] += copy.deepcopy(model_.module.spatial_mats[seen_tasks[-2]])
        model_.module.temporal_mats[seen_tasks[-1]] += copy.deepcopy(model_.module.temporal_mats[seen_tasks[-2]])
    return 


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    net = ASS_JRG(whole_size=800, patch_size=800, seg_num=12, joint_num=17, out_dim=1,save_graph=True, G_E_graph=False)
    net = net.cuda()
    print(net)
    feat_whole = torch.randn(2,12,800).cuda()
    feat_patch = torch.randn(2,17, 12,800).cuda()
    acls = 1
    fused_feat, cell0 = net(feat_whole, feat_patch)
    print(fused_feat.shape)
    print(cell0[0].shape)