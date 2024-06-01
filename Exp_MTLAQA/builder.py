import os,sys
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm
import pickle
# from models.JRG_ASS import ASS_JRG
from models.LSTM import LSTM_anno
from models.MLP import MLP_block
import torch.utils.data as Data
# from seven_dataset import Seven_Dataset
import argparse
import random
from torchvideotransforms import video_transforms, volume_transforms


# ======================================================== 
def build_moodel(args):
    # model_ = ASS_JRG(whole_size=args.whole_size, patch_size=args.patch_size, seg_num=args.seg_num, joint_num=args.joint_num, out_dim=1,save_graph=args.save_graph, mode = args.mode, G_E_graph=args.g_e_graph, alpha=args.alpha)
    lstm = LSTM_anno()
    score_rgs = MLP_block(512, 1)
    diff_rgs = MLP_block(1024, 1)
    return lstm, score_rgs, diff_rgs

def load_ckpt():
    return

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_data(data_root, set_id, args, exemplar_set=None, multi_actions=False, num_tasks=2):
    if num_tasks not in [2,4]:
        print('number of tasks should be in [2,4]')
        assert num_tasks in [2,4]

    feat_file = 'YOUR_DATA_ROOT/MTL-AQA/feature/Kinetics400_i3d_MTL_AQA_seg_10_dim_1024.pkl'
    name_file = 'YOUR_DATA_ROOT/MTL-AQA/feature/MTL_video_names.pkl'
    with open(feat_file, 'rb') as f:
        video_dict = pickle.load(f)
    with open(name_file, 'rb') as f:
        video_names = pickle.load(f)

    train_all_feature = video_dict['train_feat']
    train_all_score = video_dict['train_final_score']
    train_all_name = video_names['train_video_name']

    test_all_feature = video_dict['test_feat']
    test_all_score = video_dict['test_final_score']
    test_all_name = video_names['test_video_name']

    if num_tasks == 2:
        competition_sets = [[1,2,3,4,5,6,7,9,10,13,14],
            [17,22,18]]
    elif num_tasks == 4:
        competition_sets = [[14,13,4,5,6,9,10], [1,2,3,7], [17, 18], [22]]
    competition_set = competition_sets[set_id]
    if multi_actions:
        competition_set = [1,2,3,4,5,6,7,9,10,13,14,17,22,18]
    train_feats = torch.Tensor([])
    train_scores = torch.Tensor([])
    train_names = torch.Tensor([])
    test_feats = torch.Tensor([])
    test_scores = torch.Tensor([])
    test_names = torch.Tensor([])

    for i in range(len(train_all_score)):
        if train_all_name[i][0] not in competition_set:
            continue
        train_feats = torch.cat([train_feats, train_all_feature[i].unsqueeze(0)], dim=0)
        train_scores = torch.cat([train_scores, train_all_score[i].unsqueeze(0)],dim=0)
        train_names = torch.cat([train_names, train_all_name[i].unsqueeze(0)],dim=0)
    
    for i in range(len(test_all_score)):
        if test_all_name[i][0] not in competition_set:
            continue
        test_feats = torch.cat([test_feats, test_all_feature[i].unsqueeze(0)], dim=0)
        test_scores = torch.cat([test_scores, test_all_score[i].unsqueeze(0)],dim=0)
        test_names = torch.cat([test_names, test_all_name[i].unsqueeze(0)],dim=0)

    # Score Normalize
    train_max =  np.max(train_scores.numpy())
    train_min = np.min(train_scores.numpy())
    train_scores = (train_scores - train_min) / (train_max - train_min) * 10.0
    test_scores = (test_scores - train_min) / (train_max - train_min) * 10.0
    # print('packing')

    if (exemplar_set is not None) and (args.dataset_mixup):
        for a in range(len(exemplar_set)):
            if len(exemplar_set[a]) == 0:
                continue
            for exemplar in exemplar_set[a]:
                # print(exemplar[0].shape)
                train_feats = torch.cat((train_feats, exemplar[0].float().unsqueeze(0)),dim=0)
                train_scores = torch.cat((train_scores, exemplar[1].float().unsqueeze(0)),dim=0)
                train_names = torch.cat((train_names, exemplar[2].float().unsqueeze(0)),dim=0)
                
                
    # print(len(train_scores))
    # print(len(test_scores))
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset = Data.TensorDataset( \
                    train_feats.float(), \
                    train_scores.float(),
                    train_names)
    test_dataset = Data.TensorDataset( \
                    test_feats.float(), \
                    test_scores.float(),
                    test_names)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn,generator=g)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, (train_feats, train_scores, train_names)

def load_data_from_list(data_list=None, cls_label=0):
    train_whole_with_memory = torch.Tensor([])
    # train_patch_with_memory = torch.Tensor([])
    train_scores_with_memory = torch.Tensor([])
    train_action_name_with_memory = torch.Tensor([])
    data_idx = []

    if data_list is not None:

        for idx, exemplar in enumerate(data_list):
            # train_whole_with_memory.append(exemplar[0])          
            # train_patch_with_memory.append(exemplar[1])
            train_whole_with_memory = torch.cat((train_whole_with_memory, exemplar[0].float().unsqueeze(0)),dim=0)
            # np.append(train_scores_with_memory, [exemplar[2]],axis=0)
            train_scores_with_memory = torch.cat((train_scores_with_memory, exemplar[1].float().unsqueeze(0)),dim=0)
            train_action_name_with_memory = torch.cat((train_action_name_with_memory, exemplar[2].float().unsqueeze(0)),dim=0)
            data_idx += [idx]

    dataset_train = Data.TensorDataset( \
                    train_whole_with_memory, \
                    train_scores_with_memory, 
                    train_action_name_with_memory,
                    torch.tensor(np.array(data_idx)))
    loader_train = Data.DataLoader(dataset=dataset_train, batch_size=16, shuffle=False, num_workers=0)
    print('pack finished')
    return loader_train

def load_exemplar_data(exemplar_set=[]):
    train_whole_with_memory = torch.Tensor([])
    train_scores_with_memory = torch.Tensor([])
    train_action_name_with_memory = []
    if exemplar_set is not None:
        for a in range(len(exemplar_set)):
            if len(exemplar_set[a]) == 0:
                continue
            for exemplar in exemplar_set[a]:
                # train_whole_with_memory.append(exemplar[0])          
                # train_patch_with_memory.append(exemplar[1])
                train_whole_with_memory = torch.cat((train_whole_with_memory, torch.tensor(exemplar[0]).float().unsqueeze(0)),dim=0)
                train_patch_with_memory = torch.cat((train_patch_with_memory, torch.tensor(exemplar[1]).float().unsqueeze(0)),dim=0)
                # np.append(train_scores_with_memory, [exemplar[2]],axis=0)
                train_scores_with_memory = torch.cat((train_scores_with_memory, torch.tensor(exemplar[2]).float().unsqueeze(0)),dim=0)
                train_action_name_with_memory += [a]
    
    dataset_train = Data.TensorDataset( \
                    train_whole_with_memory, \
                    train_patch_with_memory, \
                    train_scores_with_memory, 
                    torch.tensor(np.array(train_action_name_with_memory)))
    loader_train = Data.DataLoader(dataset=dataset_train, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    print('pack finished')
    return loader_train

def get_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--model_path', type=str, default='Model_1', help='one model per file')
    parser.add_argument('--gpu', type=str, default='2', help='id of gpu device(s) to be used')
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--seed', type=int, default=0, help='number of training epochs')
    
    # # Approaches and Ablation Study 消融实验
    parser.add_argument('--lr_decay', action='store_true', default=False, help='Learning rate decay')  # 是否进行学习率衰减
    parser.add_argument('--graph_distill', action='store_true', default=False, help='graph_distillation')  # 是否进行graph distill
    parser.add_argument('--replay', action='store_true', default=False, help='replay')  # 是否replay
    parser.add_argument('--pod_loss', action='store_true', default=False, help='podloss')  # 是否加入podloss
    parser.add_argument('--diff_loss', action='store_true', default=False, help='diffloss')  # 是否加入podloss
    parser.add_argument('--aug_rgs', action='store_true', default=False, help='aug-rgs')
    parser.add_argument('--save_ckpt', action='store_true', default=False, help='save ckpt') # 是否存ckpt
    parser.add_argument('--save_graph', action='store_true', default=False, help='save graph')
    parser.add_argument('--g_e_graph', action='store_true', default=False, help='general_expert_graph')
    parser.add_argument('--graph_visualization', action='store_true', default=False, help='visualize graphs') # 是否存ckpt
    parser.add_argument('--upper_bound', action='store_true', default=False, help='upper bound') # 是否是在跑upper bound

    parser.add_argument('--dataset_mixup', action='store_true', default=False, 
        help='If set as True, previous data will mix with current data during training. If --aug_approach!=None, --dataset_mixup must set as False') # 是否存ckpt

    parser.add_argument('--approach', type=str, default='finetune', \
        choices=['finetune', 'distill', 'lwf', 'ewc', 'group_replay', 'random_replay', 'herding_replay','podnet', 'aug-diff', 'e_graph', 'g_e_graph'])
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
    parser.add_argument('--num_tasks', type=int, default=2, choices=[2, 4], help='number of tasks (2 or 4)') # task 的数量



    # Hyper-parameters 超参数
    parser.add_argument('--visualization_schedule', type=int, default=10, help='visualization_schedule')
    parser.add_argument('--num_helpers', type=int, default=3, help='number of training epochs')
    parser.add_argument('--aug_scale', type=float, default=0.3, help='number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--base_lr', type=float, default=0.001, help='basic learning rate')
    parser.add_argument('--lr_factor', type=float, default=1, help='lr_factor')
    parser.add_argument('--memory_size', type=int, default=60, help='memory size')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for GE_graph')
    parser.add_argument('--lambda_distill', type=float, default=7, help='lambda_d')
    parser.add_argument('--lambda_diff', type=float, default=1, help='lambda_diff')
    parser.add_argument('--lambda_graph_distill', type=float, default=7, help='lambda_graph_distill')
    parser.add_argument('--lambda_pod', type=float, default=1, help='lambda_graph_distill')
    parser.add_argument('--lambda_ewc', type=float, default=1, help='lambda_ewc')

    parser.add_argument('--optim_id', type=int, default=1, help='optimizer id ')
    
    # Batchsize and epochs
    parser.add_argument('--num_epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=32, help='number of subprocesses for dataloader')
    parser.add_argument('--batch-size', type=int, default=32)           # 原文batch size是64


    # Dataset & Model
    parser.add_argument('-tf-or-torch', type=str, default='torch')
    parser.add_argument('-kinetics-or-charades', type=str, default='kinetics')
    parser.add_argument('-swind-or-segm', type=str, default='swind')
    parser.add_argument('--data_root', type=str, default='/data1/yuanming/AQA7')
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

def check_args_valid(args):
    if args.aug_approach != 'none':
        assert (not args.dataset_mixup)
    if args.aug_rgs:
        assert (not args.diff_loss)
    return

def build_exp():
    args = get_parser().parse_args()
    # set gpu
    check_args_valid(args)
    print('gpu: ', args.gpu)
    
    # # set seeds
    seed = args.seed
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    if args.num_tasks == 2:
        classes_name = ['diving-10m', 'diving-3m']
        task_list_bank = [[0,1],
                [1,0]]
    elif args.num_tasks == 4:
        classes_name = ['W-10m','M-10m','M-3m' ,'W-3m']
        task_list_bank = [[2,3,1,0], [3,2,0,1], [0,1,2,3], [3,1,0,2]]
    if seed in [0,1]:
        task_list = task_list_bank[seed]
    else:
        task_list = task_list_bank[1]
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
