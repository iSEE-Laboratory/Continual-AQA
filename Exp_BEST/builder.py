import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
import time
from models.JRG_ASS_AGSG import ASS_JRG
from models.MLP import MLP_block

import argparse
import random

def build_moodel(args):
    # model_ = ASS_JRG(whole_size=args.whole_size, patch_size=args.patch_size, seg_num=args.seg_num, joint_num=args.joint_num, out_dim=1,save_graph=args.save_graph, mode = args.mode, G_E_graph=args.g_e_graph, alpha=args.alpha)
    model_ = ASS_JRG(whole_size=1024, mode='identity', is_map=False, afc=args.afc, save_graph=args.save_graph, g_e_graph=args.g_e_graph, alpha=args.alpha)
    score_rgs = MLP_block(512, 1)
    diff_rgs = MLP_block(1024, 1)
    return model_, score_rgs, diff_rgs


def load_data(set_id, split_root, feature_root, batch_size=16, exemplar_set=None, args=None):
    act_names = ['apply_eyeliner','braid_hair','origami','scrambled_eggs','tie_tie']

    annot_train = os.path.join(split_root, act_names[set_id], 'train.txt')
    annot_test = os.path.join(split_root, act_names[set_id], 'test.txt')

    train_all_feat_dict = {}
    test_all_feat_dict = {}
    train_whole1 = []
    train_whole2 = []
    test_whole1 = []
    test_whole2 = []
    all_pair = 0
    # load training data 
    for line in open(annot_train,'r'):
        names =  re.split(' |\n',line)
        name1 = names[0]
        name2 = names[1]
        all_pair += 1
        if name1 not in train_all_feat_dict.keys():
            file_name1 = os.path.join(feature_root, act_names[set_id], name1+'_rgb.npz')
            train_all_feat_dict[name1] = np.load(file_name1)['arr_0']
        if name2 not in train_all_feat_dict.keys():
            file_name2 = os.path.join(feature_root, act_names[set_id], name2+'_rgb.npz')
            train_all_feat_dict[name2] = np.load(file_name2)['arr_0']
        train_whole1.append(train_all_feat_dict[name1])
        train_whole2.append(train_all_feat_dict[name2])
    # load testing data
    for line in open(annot_test,'r'):
        names =  re.split(' |\n',line)
        name1 = names[0]
        name2 = names[1]
        all_pair += 1
        if name1 not in test_all_feat_dict.keys():
            file_name1 = os.path.join(feature_root, act_names[set_id], name1+'_rgb.npz')
            test_all_feat_dict[name1] = np.load(file_name1)['arr_0']
        if name2 not in test_all_feat_dict.keys():
            file_name2 = os.path.join(feature_root, act_names[set_id], name2+'_rgb.npz')
            test_all_feat_dict[name2] = np.load(file_name2)['arr_0']
        test_whole1.append(test_all_feat_dict[name1])
        test_whole2.append(test_all_feat_dict[name2])
    train_whole1 = np.array(train_whole1)
    train_whole2 = np.array(train_whole2)
    test_whole1 = np.array(test_whole1)
    test_whole2 = np.array(test_whole2)
    train_action_id_1 = [set_id for _ in range(train_whole1.shape[0])]
    train_action_id_2 = [set_id for _ in range(train_whole2.shape[0])]
    test_action_id_1 = [set_id for _ in range(test_whole1.shape[0])]
    test_action_id_2 = [set_id for _ in range(test_whole2.shape[0])]

    # insert exemplars
    exemplar_feat_list = torch.Tensor([])
    exemplar_score_list = torch.Tensor([])
    exemplar_action_id_list = []
    empty_count = 0
    if (exemplar_set is not None) and (args.dataset_mixup):
        for action in range(len(exemplar_set)):
            if len(exemplar_set[action]) == 0:
                empty_count += 1
                continue
            exemplar_feat_score = exemplar_set[action]
            # print(exemplar_feat_score[0][0].shape)     # [400, 1024]
            # print(exemplar_feat_score[0][1].shape)     # []
            exemplar_feat_one_action = torch.cat([f_s[0].unsqueeze(0) for f_s in exemplar_feat_score])
            exemplar_score_one_action = torch.cat([f_s[1].unsqueeze(0) for f_s in exemplar_feat_score])
            exemplar_feat_list = torch.cat([exemplar_feat_list, exemplar_feat_one_action], dim=0)
            exemplar_score_list = torch.cat([exemplar_score_list, exemplar_score_one_action], dim=0)
            exemplar_action_id_list += [action for _ in range(len(exemplar_set[action]))]

        # constrcut the pair-wise ranking tasks by the exemplar set
        if empty_count != len(exemplar_set):
            # print(exemplar_feat_list.shape)
            # print(exemplar_score_list.shape)
            exemplar_feat_list_prime = torch.cat((exemplar_feat_list[1:], exemplar_feat_list[0].unsqueeze(0)), dim=0)
            exemplar_score_list_prime = torch.cat((exemplar_score_list[1:], exemplar_score_list[0].unsqueeze(0)), dim=0)
            exemplar_action_id_list_prime = exemplar_action_id_list[1:] + [exemplar_action_id_list[0]]

            exemplar_whole1 = torch.Tensor([])
            exemplar_whole2 = torch.Tensor([])
            exemplar_action_id_1 = []
            exemplar_action_id_2 = []
            for i in range(len(exemplar_score_list)):
                if exemplar_score_list[i] >= exemplar_score_list_prime[i]:
                    exemplar_whole1 = torch.cat([exemplar_whole1, exemplar_feat_list[i].unsqueeze(0)])
                    exemplar_whole2 = torch.cat([exemplar_whole2, exemplar_feat_list_prime[i].unsqueeze(0)])
                    exemplar_action_id_1 += [exemplar_action_id_list[i]]
                    exemplar_action_id_2 += [exemplar_action_id_list_prime[i]]
                else:
                    exemplar_whole1 = torch.cat([exemplar_whole1, exemplar_feat_list_prime[i].unsqueeze(0)])
                    exemplar_whole2 = torch.cat([exemplar_whole2, exemplar_feat_list[i].unsqueeze(0)])
                    exemplar_action_id_1 += [exemplar_action_id_list_prime[i]]
                    exemplar_action_id_2 += [exemplar_action_id_list[i]]
                # exemplar_whole1
            train_whole1 = torch.cat([torch.Tensor(train_whole1).float(), exemplar_whole1])
            train_whole2 = torch.cat([torch.Tensor(train_whole2).float(), exemplar_whole2])
            train_action_id_1 += exemplar_action_id_1
            train_action_id_2 += exemplar_action_id_2
            # print('train whole1 shape', train_whole1.shape)
            # print('train whole2 shape', train_whole2.shape)
            # assert 1==2
        else:
            print('exemplar set is empty')

    # get loaders
    if args is not None:
        workers = args.num_workers
    else:
        workers = 2
    # print(torch.tensor(train_whole1).float().shape)
    dataset_train=(Data.TensorDataset( \
                    torch.tensor(train_whole1).float(), \
                    torch.tensor(train_whole2).float(), \
                    torch.tensor(np.array(train_action_id_1)),\
                    torch.tensor(np.array(train_action_id_2))))
    dataset_test=(Data.TensorDataset( \
                    torch.tensor(test_whole1).float(), \
                    torch.tensor(test_whole2).float(), \
                    torch.tensor(np.array(test_action_id_1)),\
                    torch.tensor(np.array(test_action_id_2))))
    loader_train=(Data.DataLoader(\
        dataset=dataset_train, batch_size=batch_size, \
        shuffle=True, num_workers=workers))
    loader_test=(Data.DataLoader(\
        dataset=dataset_test, batch_size=8, \
        shuffle=False, num_workers=workers))

    all_train_feat_list = torch.Tensor([])
    all_test_feat_list  = torch.Tensor([])
    for key in train_all_feat_dict.keys():
        all_train_feat_list = torch.cat([all_train_feat_list, torch.tensor(train_all_feat_dict[key]).float().unsqueeze(0)], dim=0)
    for key in test_all_feat_dict.keys():
        all_test_feat_list = torch.cat([all_test_feat_list, torch.tensor(test_all_feat_dict[key]).float().unsqueeze(0)], dim=0)  
    # print(all_train_feat_list.shape)
    return loader_train, loader_test, all_train_feat_list, all_test_feat_list

def load_data_upper_bound_(split_root, feature_root, batch_size=16, args=None):
    act_names = ['apply_eyeliner','braid_hair','origami','scrambled_eggs','tie_tie']

    all_training_whole1 = torch.Tensor([])
    all_training_whole2 = torch.Tensor([])
    all_testing_whole1 = torch.Tensor([])
    all_testing_whole2 = torch.Tensor([])
    
    for set_id in range(len(act_names)):
        print('action: ', act_names[set_id])
        annot_train = os.path.join(split_root, act_names[set_id], 'train.txt')
        annot_test = os.path.join(split_root, act_names[set_id], 'test.txt')


        train_all_feat_dict = {}
        test_all_feat_dict = {}
        train_whole1 = []
        train_whole2 = []
        test_whole1 = []
        test_whole2 = []
        all_pair = 0
        # load training data 
        for line in open(annot_train,'r'):
            names =  re.split(' |\n',line)
            name1 = names[0]
            name2 = names[1]
            all_pair += 1
            if name1 not in train_all_feat_dict.keys():
                file_name1 = os.path.join(feature_root, act_names[set_id], name1+'_rgb.npz')
                train_all_feat_dict[name1] = np.load(file_name1)['arr_0']
            if name2 not in train_all_feat_dict.keys():
                file_name2 = os.path.join(feature_root, act_names[set_id], name2+'_rgb.npz')
                train_all_feat_dict[name2] = np.load(file_name2)['arr_0']
            train_whole1.append(train_all_feat_dict[name1])
            train_whole2.append(train_all_feat_dict[name2])
        # load testing data
        for line in open(annot_test,'r'):
            names =  re.split(' |\n',line)
            name1 = names[0]
            name2 = names[1]
            all_pair += 1
            if name1 not in test_all_feat_dict.keys():
                file_name1 = os.path.join(feature_root, act_names[set_id], name1+'_rgb.npz')
                test_all_feat_dict[name1] = np.load(file_name1)['arr_0']
            if name2 not in test_all_feat_dict.keys():
                file_name2 = os.path.join(feature_root, act_names[set_id], name2+'_rgb.npz')
                test_all_feat_dict[name2] = np.load(file_name2)['arr_0']
            test_whole1.append(test_all_feat_dict[name1])
            test_whole2.append(test_all_feat_dict[name2])
        train_whole1 = torch.Tensor(np.array(train_whole1)).float()
        train_whole2 = torch.Tensor(np.array(train_whole2)).float()
        test_whole1 = torch.Tensor(np.array(test_whole1)).float()
        test_whole2 = torch.Tensor(np.array(test_whole2)).float()
        print(train_whole1.shape)
        print(train_whole2.shape)
        print(test_whole1.shape)
        print(test_whole2.shape)
        all_training_whole1 = torch.cat([all_training_whole1, train_whole1], dim=0)
        all_training_whole2 = torch.cat([all_training_whole2, train_whole2], dim=0)
        all_testing_whole1 = torch.cat([all_testing_whole1, test_whole1], dim=0)
        all_testing_whole2 = torch.cat([all_testing_whole2, test_whole2], dim=0)

    print(all_training_whole1.shape)
    print(all_training_whole2.shape)
    print(all_testing_whole1.shape)
    print(all_testing_whole2.shape)
    # get loaders
    if args is not None:
        workers = args.num_workers
    else:
        workers = 2
    # print(torch.tensor(train_whole1).float().shape)
    dataset_train=(Data.TensorDataset( \
                    all_training_whole1, \
                    all_training_whole2.float()))
    dataset_test=(Data.TensorDataset( \
                    all_testing_whole1, \
                    all_testing_whole2))
    loader_train=(Data.DataLoader(\
        dataset=dataset_train, batch_size=batch_size, \
        shuffle=True, num_workers=workers))
    loader_test=(Data.DataLoader(\
        dataset=dataset_test, batch_size=batch_size, \
        shuffle=False, num_workers=workers))
    return loader_train, loader_test


def build_loader_from_list_(task, feat_list, batch_size=2, args=None):
    '''
        feat_list is a Tensor (n, 400, 1024)
    '''
    data_idx = np.arange(feat_list.shape[0])
    action_id = np.array([task] * len(feat_list))
    dataset = Data.TensorDataset(feat_list, torch.DoubleTensor(data_idx), torch.tensor(action_id))
    if args is not None:
        workers = args.num_workers
    else:
        workers = 2
    loader=Data.DataLoader(\
        dataset=dataset, batch_size=batch_size, \
        shuffle=False, num_workers=workers)
    return loader

def get_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--model_path', type=str, default='Model_1', help='one model per file')
    parser.add_argument('--gpu', type=str, default='2', help='id of gpu device(s) to be used')
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--seed', type=int, default=0, help='number of training epochs')
    
    # # Approaches and Ablation Study 消融实验
    parser.add_argument('--lr_decay', action='store_true', default=False, help='Learning rate decay')  # 是否进行学习率衰减
    parser.add_argument('--graph_distill', action='store_true', default=False, help='graph_distillation')  # 是否进行graph distill
    parser.add_argument('--feat_distill', action='store_true', default=False, help='graph_distillation')  # 是否进行feature distill
    parser.add_argument('--replay', action='store_true', default=False, help='replay')  # 是否replay
    parser.add_argument('--pod_loss', action='store_true', default=False, help='podloss')  # 是否加入podloss
    parser.add_argument('--diff_loss', action='store_true', default=False, help='diffloss')  # 是否加入diff loss
    parser.add_argument('--afc', action='store_true', default=False, help='afc')  # 是否加入afc loss
    parser.add_argument('--ewc', action='store_true', default=False, help='ewc')  # 是否加入ewc loss
    parser.add_argument('--aug_rgs', action='store_true', default=False, help='aug-rgs')
    parser.add_argument('--save_ckpt', action='store_true', default=False, help='save ckpt') # 是否存ckpt
    parser.add_argument('--save_graph', action='store_true', default=False, help='save graph')
    parser.add_argument('--g_e_graph', action='store_true', default=False, help='general_expert_graph')
    parser.add_argument('--graph_visualization', action='store_true', default=False, help='visualize graphs') # 是否存ckpt

    parser.add_argument('--dataset_mixup', action='store_true', default=False, 
        help='If set as True, previous data will mix with current data during training. If --aug_approach!=None, --dataset_mixup must set as False') # 是否存ckpt

    parser.add_argument('--approach', type=str, default='finetune', \
        choices=['finetune', 'distill', 'lwf', 'group_replay', 'random_replay', 'herding_replay','podnet', 'aug-diff', 'e_graph', 'g_e_graph'])
    parser.add_argument('--aug_approach', type=str, default='none', \
        choices=['none', 'p-distill', 'aug-diff', 'aug-rgs'])
    parser.add_argument('--aug_mode', type=str, default='fs_aug', \
        choices=['f_aug', 's_aug', 'fs_aug'], help='augmentation setting')
    parser.add_argument('--fix_graph_mode', type=str, default='fix_old', \
        choices=['fix_old', 'fix_new', 'no_fix', 'all_fix'])
    parser.add_argument('--optim_mode', type=str, default='default', \
        choices=['default', 'new_optim'])
    parser.add_argument('--replay_method', type=str, default='random', \
        choices=['random', 'herding', 'group_replay'])


    # Hyper-parameters 超参数
    parser.add_argument('--visualization_schedule', type=int, default=10, help='visualization_schedule')
    parser.add_argument('--num_helpers', type=int, default=3, help='number of training epochs')
    parser.add_argument('--aug_scale', type=float, default=0.3, help='number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='basic learning rate')
    parser.add_argument('--lr_factor', type=float, default=1, help='lr_factor')
    parser.add_argument('--memory_size', type=int, default=60, help='memory size')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for GE_graph')
    parser.add_argument('--lambda_distill', type=float, default=7, help='lambda_d')
    parser.add_argument('--lambda_diff', type=float, default=1, help='lambda_diff')
    parser.add_argument('--lambda_graph_distill', type=float, default=7, help='lambda_graph_distill')
    parser.add_argument('--lambda_pod', type=float, default=1, help='lambda_pod')
    parser.add_argument('--lambda_afc', type=float, default=1, help='lambda_afc')
    parser.add_argument('--lambda_ewc', type=float, default=1, help='lambda_ewc')
    parser.add_argument('--gama', type=float, default=0.1, help='lambda_graph_distill')

    parser.add_argument('--optim_id', type=int, default=3, help='optimizer id ')
    
    # Batchsize and epochs
    parser.add_argument('--num_epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='number of subprocesses for dataloader')
    parser.add_argument('--num_tasks', type=int, default=5, help='number of tasks')
    parser.add_argument('--batch-size', type=int, default=32)           # 原文batch size是64


    # Dataset & Model
    parser.add_argument('-tf-or-torch', type=str, default='torch')
    parser.add_argument('-kinetics-or-charades', type=str, default='kinetics')
    parser.add_argument('-swind-or-segm', type=str, default='swind')
    parser.add_argument('--split_root', type=str, default='/home/share3/BEST/splits')
    parser.add_argument('--feature_root', type=str, default='/home/share3/BEST/feature')
    parser.add_argument('--ckpt_root', type=str, default='./ckpt/')
    parser.add_argument('--ckpt_path', type=str, default='')
    
    parser.add_argument('--seg-num', type=int, default=12, help='Number of Video Segments')
    parser.add_argument('--joint-num', type=int, default=17, help='Number of human joints')
    parser.add_argument('--whole-size', type=int, default=800, help='I3D feat size')
    parser.add_argument('--patch-size', type=int, default=800, help='I3D feat size')
    
    # 暂时用不上的参数

    parser.add_argument('--model_name', type=str, default='I3D_MLP', help='name of model')

    parser.add_argument("--pretrained_i3d_weight", type=str,
                        default='../pretrained_models/i3d_model_rgb.pth',
                        help='pretrained i3d model')
    parser.add_argument('-mode', type=str, default='single-head',\
        choices=['single-head', 'multi-head'])
    return parser

def build_exp():
    args = get_parser().parse_args()
    # set gpu
    # check_args_valid(args)
    print('gpu: ', args.gpu)
    # classes_name = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
    classes_name = ['apply_eyeliner','braid_hair','origami','scrambled_eggs','tie_tie']

    # # set seeds
    seed = args.seed

    task_list_bank = [[2, 0, 1, 3, 4],
             [2, 1, 4, 0, 3],
             [2, 4, 1, 3, 0],
             [3, 4, 1, 0, 2]]
    task_list = task_list_bank[seed]
    action2task = [0] * args.num_tasks
    for i in range(args.num_tasks):
        action2task[task_list[i]] = i
    # set where to save exp results
    pre_task = -1
    print('task list: ', task_list)

    # 创建实验路径
    if not os.path.isdir(os.path.join('./ckpt', args.exp_name)):
        os.makedirs(os.path.join('./ckpt', args.exp_name))
    # 记录实验设定
    argsDict = args.__dict__
    with open(os.path.join('./ckpt', args.exp_name, 'config.yaml'), 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

    return args, task_list, action2task, classes_name

if __name__ == '__main__':
    loader_train, loader_test, all_train_feat_list, all_test_feat_list = load_data(set_id=0, split_root='/home/share3/BEST/splits', feature_root='/home/share3/BEST/feature')
    st = time.time()
    for (video1, video2) in loader_train:
        continue
    ed = time.time()
    print(ed-st)
    print()