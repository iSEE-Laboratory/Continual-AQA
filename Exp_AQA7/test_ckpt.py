import sys,time
import os
import random
import copy
from models.JRG_ASS import ASS_JRG
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as Data
# dataset
# from seven_dataset import Seven_Dataset
# model_runner
from models.JRG_ASS import run_jrg
from scipy import stats
import builder
import utils
import loss_fn
from models.MLP import MLP_block
from turtle import color
from sklearn import datasets
from sklearn.manifold import TSNE
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('agg')


def load_data(data_root, set_id):
    feat_file = os.path.join(data_root, 'AQA_pytorch_' + 'kinetics' + '_' + 'swind' + '_Set_' + str(set_id + 1) + '_Feats.npz')

    
    all_dict = np.load(feat_file)	
    print('load finished')

    train_whole = np.concatenate((all_dict['train_rgb'][:,:,0,:], all_dict['train_flow'][:,:,0,:]), axis=2)
    train_patch = np.concatenate((all_dict['train_rgb'][:,:,1::,:].transpose((0,2,1,3)),all_dict['train_flow'][:,:,1::,:].transpose((0,2,1,3))), axis=3)
    test_whole = np.concatenate((all_dict['test_rgb'][:,:,0,:],all_dict['test_flow'][:,:,0,:]), axis=2)
    test_patch = np.concatenate((all_dict['test_rgb'][:,:,1::,:].transpose((0,2,1,3)),all_dict['test_flow'][:,:,1::,:].transpose((0,2,1,3))), axis=3)

    # 读标签
    train_scores_ = all_dict['train_label']
    # train_scores_ += np.random.normal(0, args.sigma, train_scores_.shape[0]) * (np.max(np.array(train_scores_)) - np.min(np.array(train_scores_)))
    train_scores = np.repeat(train_scores_, 2)
    test_scores = all_dict['test_label']
    seg_num = test_whole.shape[1]
    feat_size = test_whole.shape[2]
    whole_size = patch_size = feat_size
    print('read finished')

    # Score Normalize
    train_max =  np.max(train_scores)
    train_min = np.min(train_scores)
    train_scores = (train_scores - train_min) / (train_max - train_min) * 10.0
    test_scores = (test_scores - train_min) / (train_max - train_min) * 10.0
    print('packing')

    train_action_name = [set_id for _ in range(train_scores.shape[0])]
    test_action_name = [set_id for _ in range(test_scores.shape[0])]
    # print(len(train_action_name))
    train_whole_with_memory  = torch.tensor(train_whole).float()
    train_patch_with_memory = torch.tensor(train_patch).float()
    train_scores_with_memory = torch.tensor(train_scores).float()
    train_action_name_with_memory = train_action_name


    # 构建 dataloader
    dataset_train = Data.TensorDataset(
                    train_whole_with_memory, \
                    train_patch_with_memory, \
                    train_scores_with_memory, 
                    torch.tensor(np.array(train_action_name_with_memory)),
                    torch.tensor(np.array([i for i in range(train_scores_with_memory.shape[0])])))
    dataset_test = Data.TensorDataset( 
                    torch.tensor(test_whole).float(), \
                    torch.tensor(test_patch).float(), \
                    torch.tensor(test_scores).float(),
                    torch.tensor(np.array([set_id for _ in range(test_scores.shape[0])]))
                    # torch.tensor(np.array([i for i in range(test_scores.shape[0])]))
                    )
    loader_test = Data.DataLoader(dataset=dataset_test, batch_size=16, shuffle=False, num_workers=2)
    print('pack finished')
    return loader_test


def test_net(t, jrg, score_rgs, dataloaders, rho_matrix, rl2_matrix, args, seen_tasks=[]):
    jrg.eval()
    score_rgs.eval()
    torch.set_grad_enabled(False)
    
    # rho_matrix=np.zeros((len(dataloaders),len(dataloaders)),dtype=np.float32)
    # rl2_matrix=np.zeros((len(dataloaders),len(dataloaders)),dtype=np.float32)
    for i, a_task in enumerate(seen_tasks):
        dataloader = dataloaders[a_task]
        true_scores = []
        pred_scores = []
        for batch_idx, (feat_whole, feat_patch, scores, action_id) in enumerate(dataloader):
            # print('step {}'.format(step))
            feat_whole = feat_whole.cuda()
            feat_patch = feat_patch.cuda()
            scores = scores.float().cuda()
            action_id = action_id.type(torch.int64).cuda()
            batch_size = action_id.shape[0]

            feat, _ = run_jrg(jrg, feat_whole, feat_patch, save_graph=True, seen_tasks=seen_tasks, args=args)
            if True:
                feat = feat.transpose(0,1).gather(1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,512)).squeeze(1) 
            pred_score = score_rgs(feat)

            pred_scores.extend(pred_score.detach().data.cpu().reshape(-1).numpy())
            true_scores.extend(scores.detach().data.cpu().reshape(-1).numpy())

        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
            true_scores.shape[0]
        rho_matrix[i][t] = rho
        rl2_matrix[i][t] = RL2
    return rho_matrix, rl2_matrix

def main(ckpt_path, t=5 ,task_list=[3, 5, 4, 1, 0, 2]):
    # load data
    loaders_test = []
    start_time = time.time()
    for i in range(len(task_list)):
        loader_test = load_data(data_root='/home/share3/AQA-7/feature', set_id=i)
        loaders_test.append(loader_test)
    # load model
    jrg = ASS_JRG(whole_size=800, patch_size=800, seg_num=12, joint_num=17, out_dim=1,save_graph=True, mode = '', G_E_graph=True, alpha=0.8)
    score_rgs = MLP_block(512, 1)
    
    jrg_state_dict = torch.load(ckpt_path, map_location='cpu')['jrg']
    rgs_state_dict = torch.load(ckpt_path, map_location='cpu')['regressor']
    # model_ckpt = {k.replace("module.", ""): v for k, v in state_dict.items()}
    jrg_ckpt = {k.replace("module.", ""): v for k, v in jrg_state_dict.items()}
    jrg.load_state_dict(jrg_ckpt)
    jrg = jrg.cuda()
    jrg = nn.DataParallel(jrg)

    rgs_ckpt = {k.replace("module.", ""): v for k, v in rgs_state_dict.items()}
    score_rgs.load_state_dict(rgs_ckpt)
    score_rgs = score_rgs.cuda()
    score_rgs = nn.DataParallel(score_rgs)

    rho_matrix=np.zeros((len(loaders_test),len(loaders_test)),dtype=np.float32)
    rl2_matrix=np.zeros((len(loaders_test),len(loaders_test)),dtype=np.float32)
    rho_matrix, rl2_matrix = test_net(t, jrg, score_rgs, loaders_test, rho_matrix, rl2_matrix, None, [3, 5, 4, 1, 0, 2])
    print('rho_matrix: ', rho_matrix)

    # we follow previous works to calculate the average performance across tasks with fisher_z score
    avg_rho = utils.fisher_z(rho_mat[:, -1])
    print('avg_rho: ', avg_rho)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    main('/home/yuanming/Code/IAQA/ablation/ckpt/exp2-new-seed_3/0_snowboard_big_air_best@45.pth')
