import pdb
import os
import numpy as np 
import scipy.stats as stats
# from sklearn import metrics
import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange, repeat, reduce
import copy

def get_loss(pred, labels, type='new_mse'):
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


def gradhook(self, grad_input, grad_output):
    # print(grad_output[0].shape)
    # print(grad_output)
    # assert 1==2
    importance = grad_output[0] ** 2 # [N, C, T, J, M]
    # print(importance.shape)
    if len(importance.shape) == 5:
        # print('here')
        importance = torch.sum(importance, 4) # [N, C, H]
        importance = torch.sum(importance, 3) # [N, C, H]
        importance = torch.sum(importance, 2) # [N, C]
    importance = torch.mean(importance, 0) # [C]
    # print(importance.shape)
    # assert 1==2
    self.importance += importance

class Channel_Importance_Measure(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.scale = nn.Parameter(torch.randn(num_channels), requires_grad=False)
        nn.init.constant_(self.scale, 1.0)
        self.register_buffer('importance', torch.zeros_like(self.scale))


    def forward(self, x):
        # print(x)
        # assert 1==2
        if len(x.shape) == 5:
            x = x * self.scale.reshape([1,-1,1,1,1])
        else:
            x = x * self.scale.reshape([1,-1])
        return x

def build_joint_graphs(joint_num, hop_num=1):
    a_file = './mat_a.npy'
    if not os.path.isfile(a_file):
        a_file = '/home/yuanming/Code/IAQA/ablation/mat_a.npy'
    if joint_num == 17:
        graph = np.load(a_file).astype(float) + np.identity(joint_num)
    else:
        graph = np.array([[1,1,0,0,0,0,0,0],
                          [1,1,1,0,0,0,0,0],
                          [0,1,1,1,0,0,0,0],
                          [0,0,1,1,1,0,0,0],
                          [0,0,0,1,1,1,0,0],
                          [0,0,0,0,1,1,1,0],
                          [0,0,0,0,0,1,1,1],
                          [0,0,0,0,0,0,1,1],
                          ])
    graph = torch.from_numpy(graph).int()
    graphs = [graph, ]
    for i in range(hop_num-1):
        tmp = torch.matmul(graphs[-1], graph)
        tmp[tmp != 0] = 1
        graphs.append(tmp)
    for i in range(hop_num-1, 0, -1):
        graphs[i] = graphs[i] * (1-graphs[i-1])
    return graphs


class ASS_JRG(nn.Module):
    def __init__(self, whole_size=1024, mode='base', is_map=False, afc=False, save_graph=False, g_e_graph=False, task_num=5, alpha=1):
        super(ASS_JRG,self).__init__()
        self.whole_size = whole_size
        self.task_num = task_num
        self.hop_num = 1
        self.alpha = alpha

        self.is_map = is_map
        if not is_map:
            self.seg_num = 10
            self.joint_num = 8
            self.module_num = 5
            self.dropout_rate = 0.2 #0.2
            self.hidden1 = 256
            self.hidden2 = 128
            self.hidden3 = 32
        else:
            self.seg_num = 20
            self.joint_num = 7
            self.module_num = 7
            self.dropout_rate = 0.2 #0.2
            self.hidden1 = 256
            self.hidden2 = 128
            self.hidden3 = 32
        #if mode == 'identity':
        #	self.module_num = 3


        # AFC
        self.afc = afc
        if self.afc:
            print('use afc')

        self.mode = mode
        self.scale = nn.Linear(1,1, bias=False).cuda()
        
        # Feature Encoders
        self.encode_diffwhole_512 = nn.Sequential(
            nn.Linear(self.whole_size, int(self.hidden1/2)),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(inplace=True),
            ).cuda()
        self.encode_whole_512 = nn.Sequential(
            nn.Linear(self.whole_size, int(self.hidden1/2)),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(inplace=True),
            ).cuda()

        # AGSG
        self.save_graph = save_graph
        self.g_e_graph = g_e_graph
        self.register_buffer("joint_graphs", torch.stack(build_joint_graphs(self.joint_num), dim=0))
        self.register_parameter("spatial_mats",
                                nn.Parameter(torch.zeros(self.task_num, self.hop_num, self.joint_num, self.joint_num)))
        self.register_parameter("temporal_mats",
                                nn.Parameter(torch.zeros(self.task_num, self.hop_num, self.joint_num, self.joint_num)))
        self.register_parameter("general_spatial_mats",
                                nn.Parameter(torch.randn(self.hop_num, self.joint_num, self.joint_num)))
        self.register_parameter("general_temporal_mats",
                                nn.Parameter(torch.randn(self.hop_num, self.joint_num, self.joint_num)))



        # Assessment Module
        self.assessment1 = nn.Sequential(
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(self.hidden1*2),
            nn.Linear(self.hidden1*2, self.hidden2 * 4),
            nn.Dropout(self.dropout_rate)).cuda()
        self.assessment2 = nn.Sequential(
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(self.hidden2),
            nn.Linear(self.hidden2, 1),
            nn.Dropout(self.dropout_rate)).cuda()

        self.last_fuse = nn.Linear(self.module_num, 1, bias=False).cuda() #nn.AdaptiveAvgPool2d((2,2)).cuda()

        self.importance = Channel_Importance_Measure(512)

    def forward(self, feat_whole):
        # print('a')
        if self.save_graph:
            # print(self.temporal_mats.shape) # 5,1,8,8
            # print(self.joint_graphs.shape)  # 1,5,5
            graphs = torch.abs(self.temporal_mats * self.joint_graphs)

        if self.g_e_graph:
            general_graphs = torch.abs(self.general_temporal_mats * self.joint_graphs)
            graphs = (1-self.alpha) * graphs + self.alpha * general_graphs
        elif not self.save_graph:
            general_graphs = torch.abs(self.general_temporal_mats * self.joint_graphs)
            graphs = general_graphs.unsqueeze(0)
        
        if not self.is_map:
            # Diff Whole
            diff_mat_Fp0 = feat_whole
            diff_mat_Fp1 = torch.cat((feat_whole[:,1:feat_whole.shape[1],:], 
                feat_whole[:,feat_whole.shape[1]-1,:].unsqueeze(1)), dim=1)
            feat_diff = torch.abs(diff_mat_Fp1 - diff_mat_Fp0)
        # print(feat_diff.shape) # B, 400, 1024
        # Encoding shape(batch, seg_num, joint_num, hidden1)
        encoded_whole = self.encode_whole_512(feat_whole).reshape(-1, self.seg_num, self.joint_num, self.module_num, int(self.hidden1/2))
        encoded_diff = self.encode_diffwhole_512(feat_diff).reshape(-1, self.seg_num, self.joint_num, self.module_num, int(self.hidden1/2))   # B, T, J, M, D
        # if self.save_graph:
        if True:
            B = len(feat_whole)
            D = encoded_whole.shape[-1]
            T = encoded_whole.shape[1]
            M = encoded_whole.shape[-2]
            encoded_whole_prime = rearrange(torch.matmul(rearrange(encoded_whole, 'B T J M D -> (B T D M) J'), graphs),
                                'A H (B T D M) J ->A H B T J M D', B=B, D=D, T=T, M=M)
            encoded_diff_prime = rearrange(torch.matmul(rearrange(encoded_diff, 'B T J M D -> (B T D M) J'), graphs),
                                'A H (B T D M) J ->A H B T J M D', B=B, D=D, T=T, M=M)
        # print(encoded_whole_prime.shape)
            if self.hop_num == 1:
                encoded_whole_prime = encoded_whole_prime[:,0] # A B T J M D
                encoded_diff_prime = encoded_diff_prime[:,0] # A B T J M D
        
        # if self.save_graph:
        if True:
            # print(encoded_whole_prime.shape)
            # print(encoded_whole.shape)
            encoded_whole_ = repeat(encoded_whole, 'B T J M D -> A B T J M D', A=encoded_whole_prime.shape[0])
            encoded_diff_ = repeat(encoded_diff, 'B T J M D -> A B T J M D', A=encoded_diff_prime.shape[0])
            CELL0 = torch.cat((encoded_whole_prime, encoded_diff_prime, encoded_whole_, encoded_diff_), dim=-1)     # A, B, 10, 8, 5, 512
        else:
            CELL0 = torch.cat((encoded_whole, encoded_diff), dim=-1)     # B, 10, 8, 5, 256

        if self.afc:
            # print('here')
            # print(CELL0.shape)
            CELL0 = CELL0[0]        # B, 10, 8, 5, 256
            fused_feat = rearrange(CELL0, 'B T J X D -> B D T J X')
            fused_feat = self.importance(fused_feat)
            fused_feat = rearrange(fused_feat, 'B D T J X -> B T J X D')
            fused_feat = fused_feat.unsqueeze(0)
        # fused_feat = reduce(fused_feat, 'B T J X D -> B D', 'mean')
            importance = [self.importance.importance]
            CELL0 = fused_feat
        else:
            importance = None
            fused_feat = CELL0 

        # print('fused feature: ', fused_feat.shape)
        fused_feat = torch.mean(torch.mean(torch.mean(fused_feat, dim=4), dim=3), dim=2)
        # print(fused_feat.shape )
        # tot_scores = self.assessment2(self.assessment1(fused_feat)).reshape(-1)
        fused_feat2 = self.assessment1(fused_feat).transpose(0,1)   # B, A, D
        # return fused_feat2, CELL0, cosine_tensor, l2_tensor.reshape(-1)
        if self.save_graph:
            CELL0 = CELL0.transpose(0,1)
        else:
            fused_feat2 = fused_feat2[:, 0]
            CELL0 = CELL0[0]
        # print('b')
        
        return fused_feat2, CELL0, None, None, importance
        # return tot_scores, None, None, None, cosine_tensor, l2_tensor.reshape(-1), None, fused_feat
		
    def start_cal_importance(self):
        # self._hook = [self.importance.register_backward_hook(gradhook)]
        self._hook = [self.importance.register_full_backward_hook(gradhook)]

    def reset_importance(self):
        self.importance.importance.zero_()

    def normalize_importance(self):
        total_importance = torch.mean(self.importance.importance)
        self.importance.importance = self.importance.importance/total_importance

    def stop_cal_importance(self):
        for hook in self._hook:
            hook.remove()
        self._hook = None
		
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

def init_e_graph(model_, t, seen_tasks=[]):
    if t==0:
        alpha = model_.module.alpha
        # g_spatial_mat = copy.deepcopy(model_.module.general_spatial_mats)
        g_temporal_mat = copy.deepcopy(model_.module.general_temporal_mats)
        # model_.module.spatial_mats[seen_tasks[0]] = g_spatial_mat
        model_.module.temporal_mats[seen_tasks[0]] = g_temporal_mat
    else:
        model_.module.temporal_mats[seen_tasks[-1]] += copy.deepcopy(model_.module.temporal_mats[seen_tasks[-2]])
    return 

if __name__ == '__main__':
    feats = torch.randn(3, 400, 1024).cuda()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # model = ASS_JRG(whole_size=1024, mode='identity', is_map=False, save_graph=True, g_e_graph=True).cuda()
    model = ASS_JRG(whole_size=1024, mode='identity', is_map=False, save_graph=False, g_e_graph=False).cuda()
    fused_feat2, CELL0, _, _, importance = model(feats.cuda())
    print(fused_feat2.shape)
    print(CELL0[0].shape)

