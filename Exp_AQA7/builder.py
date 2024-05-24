import os
import numpy as np
import torch
from models.JRG_ASS import ASS_JRG
from models.MLP import MLP_block
import torch.utils.data as Data
import argparse
import random



# ======================================================== 
def build_moodel(args):
    model_ = ASS_JRG(whole_size=args.whole_size, patch_size=args.patch_size, seg_num=args.seg_num, joint_num=args.joint_num, out_dim=1,save_graph=args.save_graph, mode = args.mode, G_E_graph=args.g_e_graph, alpha=args.alpha)
    score_rgs = MLP_block(512, 1)
    diff_rgs = MLP_block(1024, 1)
    return model_, score_rgs, diff_rgs

def load_ckpt():
    return

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_data(data_root, set_id, args, exemplar_set=None):
    feat_file = os.path.join(data_root, 'AQA_pytorch_' + 'kinetics' + '_' + 'swind' + '_Set_' + str(set_id + 1) + '_Feats.npz')

    
    all_dict = np.load(feat_file)	
    print('load finished')

    train_whole = np.concatenate((all_dict['train_rgb'][:,:,0,:], all_dict['train_flow'][:,:,0,:]), axis=2)
    train_patch = np.concatenate((all_dict['train_rgb'][:,:,1::,:].transpose((0,2,1,3)),all_dict['train_flow'][:,:,1::,:].transpose((0,2,1,3))), axis=3)
    test_whole = np.concatenate((all_dict['test_rgb'][:,:,0,:],all_dict['test_flow'][:,:,0,:]), axis=2)
    test_patch = np.concatenate((all_dict['test_rgb'][:,:,1::,:].transpose((0,2,1,3)),all_dict['test_flow'][:,:,1::,:].transpose((0,2,1,3))), axis=3)

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

    if (exemplar_set is not None) and (args.dataset_mixup):
        if 'debug' in args.exp_name:
            print('dataset_mixup')
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

    # build dataloader
    g = torch.Generator()
    g.manual_seed(args.seed)
    dataset_train = Data.TensorDataset( \
                    train_whole_with_memory, \
                    train_patch_with_memory, \
                    train_scores_with_memory, 
                    torch.tensor(np.array(train_action_name_with_memory)))
    dataset_test = Data.TensorDataset( \
                    torch.tensor(test_whole).float(), \
                    torch.tensor(test_patch).float(), \
                    torch.tensor(test_scores).float(),
                    torch.tensor(np.array([set_id for _ in range(test_scores.shape[0])])))
    loader_train = Data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn,generator=g)
    loader_test = Data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print('pack finished')
    return dataset_train, dataset_test, loader_train, loader_test, (train_whole, train_patch, train_scores)

def load_data_from_list(data_list=None, cls_label=0):
    train_whole_with_memory = torch.Tensor([])
    train_patch_with_memory = torch.Tensor([])
    train_scores_with_memory = torch.Tensor([])
    train_action_name_with_memory = []
    data_idx = []
    if data_list is not None:

        for idx, exemplar in enumerate(data_list):
            train_whole_with_memory = torch.cat((train_whole_with_memory, torch.tensor(exemplar[0]).float().unsqueeze(0)),dim=0)
            train_patch_with_memory = torch.cat((train_patch_with_memory, torch.tensor(exemplar[1]).float().unsqueeze(0)),dim=0)
            # np.append(train_scores_with_memory, [exemplar[2]],axis=0)
            train_scores_with_memory = torch.cat((train_scores_with_memory, torch.tensor(exemplar[2]).float().unsqueeze(0)),dim=0)
            train_action_name_with_memory += [cls_label]
            data_idx += [idx]


    dataset_train = Data.TensorDataset( \
                    train_whole_with_memory, \
                    train_patch_with_memory, \
                    train_scores_with_memory, 
                    torch.tensor(np.array(train_action_name_with_memory)),
                    torch.tensor(np.array(data_idx)))
    loader_train = Data.DataLoader(dataset=dataset_train, batch_size=16, shuffle=False, num_workers=0)
    print('pack finished')
    return loader_train

def load_exemplar_data(exemplar_set=[]):
    train_whole_with_memory = torch.Tensor([])
    train_patch_with_memory = torch.Tensor([])
    train_scores_with_memory = torch.Tensor([])
    train_action_name_with_memory = []
    if exemplar_set is not None:
        for a in range(len(exemplar_set)):
            if len(exemplar_set[a]) == 0:
                continue
            for exemplar in exemplar_set[a]:
                train_whole_with_memory = torch.cat((train_whole_with_memory, torch.tensor(exemplar[0]).float().unsqueeze(0)),dim=0)
                train_patch_with_memory = torch.cat((train_patch_with_memory, torch.tensor(exemplar[1]).float().unsqueeze(0)),dim=0)
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
    
    parser.add_argument('--lr_decay', action='store_true', default=False, help='Learning rate decay')  # 是否进行学习率衰减
    parser.add_argument('--graph_distill', action='store_true', default=False, help='graph_distillation')  # 是否进行graph distill
    parser.add_argument('--replay', action='store_true', default=False, help='replay')  # 是否replay
    parser.add_argument('--pod_loss', action='store_true', default=False, help='podloss')  # 是否加入podloss
    parser.add_argument('--diff_loss', action='store_true', default=False, help='diffloss')  # 是否加入podloss
    parser.add_argument('--aug_rgs', action='store_true', default=False, help='aug-rgs')
    parser.add_argument('--save_ckpt', action='store_true', default=False, help='save ckpt') # 是否存ckpt
    parser.add_argument('--save_graph', action='store_true', default=False, help='save graph')
    parser.add_argument('--aug_w_weight', action='store_true', default=False, help='augmentation with weight')
    parser.add_argument('--g_e_graph', action='store_true', default=False, help='general_expert_graph')
    parser.add_argument('--graph_visualization', action='store_true', default=False, help='visualize graphs') # 是否存ckpt
    parser.add_argument('--graph_random_init', action='store_true', default=False, help='randomly initialize grapohs') # 是随机初始化graph

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

    parser.add_argument('--visualization_schedule', type=int, default=10, help='visualization_schedule')
    parser.add_argument('--num_helpers', type=int, default=3, help='number of training epochs')
    parser.add_argument('--aug_scale', type=float, default=0.3, help='number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay')
    parser.add_argument('--base_lr', type=float, default=0.001, help='basic learning rate')
    parser.add_argument('--lr_factor', type=float, default=1, help='lr_factor')
    parser.add_argument('--memory_size', type=int, default=60, help='memory size')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for GE_graph')
    parser.add_argument('--lambda_distill', type=float, default=7, help='lambda_d')
    parser.add_argument('--lambda_diff', type=float, default=1, help='lambda_diff')
    parser.add_argument('--lambda_graph_distill', type=float, default=7, help='lambda_graph_distill')
    parser.add_argument('--lambda_pod', type=float, default=1, help='lambda_graph_distill')

    parser.add_argument('--optim_id', type=int, default=1, help='optimizer id ')
    
    parser.add_argument('--num_epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=32, help='number of subprocesses for dataloader')
    parser.add_argument('--batch-size', type=int, default=32)           # 原文batch size是64


    # Dataset & Model
    parser.add_argument('-tf-or-torch', type=str, default='torch')
    parser.add_argument('-kinetics-or-charades', type=str, default='kinetics')
    parser.add_argument('-swind-or-segm', type=str, default='swind')
    # parser.add_argument('--data_root', type=str, default='/data1/yuanming/AQA7')
    parser.add_argument('--data_root', type=str, default='/mnt/Datasets/AQA-7/feature')

    parser.add_argument('--ckpt_root', type=str, default='./ckpt/')
    parser.add_argument('--ckpt_path', type=str, default='')
    
    parser.add_argument('--seg-num', type=int, default=12, help='Number of Video Segments')
    parser.add_argument('--joint-num', type=int, default=17, help='Number of human joints')
    parser.add_argument('--whole-size', type=int, default=800, help='I3D feat size')
    parser.add_argument('--patch-size', type=int, default=800, help='I3D feat size')

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
    classes_name = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
    # # set seeds
    seed = args.seed
    
    # task_list = np.random.shuffle([i for i in range(6)])
    task_list_bank = [[5, 2, 1, 3, 0, 4],     # The task list is obtained with np.random.shuffle([i for i in range(6)])
             [2, 1, 4, 0, 3, 5],
             [4, 1, 3, 2, 5, 0],
             [3, 5, 4, 1, 0, 2]]
    task_list = task_list_bank[seed]
    
    action2task = [0] * 6
    for i in range(6):
        action2task[task_list[i]] = i
    # set where to save exp results
    pre_task = -1
    print('task list: ', task_list)


    if not os.path.isdir(os.path.join('./ckpt', args.exp_name)):
        os.makedirs(os.path.join('./ckpt', args.exp_name))

    argsDict = args.__dict__
    with open(os.path.join('./ckpt', args.exp_name, 'config.yaml'), 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
            
    return args, task_list, action2task, classes_name