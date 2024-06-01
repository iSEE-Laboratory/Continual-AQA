from builtins import print
import sys,time
import os
import random
import copy

import numpy as np
import get_optim
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
# dataset
# from seven_dataset import Seven_Dataset
# model_runner
from scipy import stats
import builder
import utils
import loss_fn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from occupy_gpu import occupy_gpu_memory


def train_epoch(t, task, epoch, lstm, lstm_pre, score_rgs, diff_rgs, dataloader, optimizer, mse, kl, ce, softmax, args, seen_tasks=[], combined_exemplar=None):
    lstm.train()
    score_rgs.train()
    diff_rgs.train()
    lstm_pre.eval()
    torch.set_grad_enabled(True)

    total_loss = 0.0
    total_mse = 0.0
    total_distill = 0.0
    total_st_pod = 0.0
    total_diff = 0.0
    total_graph_distill = 0.0
    total_ewc = 0.0

    count = 0
    for batch_idx, (feat_whole, scores, action_id) in enumerate(dataloader):
        batch_size = action_id.shape[0]
        # old data
        if combined_exemplar is not None:
            # assert 1==0

            select_whole, select_score, select_action_id, select_id = utils.random_select_exemplar(combined_exemplar, batch_size, count)

            feat_whole = torch.cat([feat_whole, select_whole], dim=0)
            # feat_patch = torch.cat([feat_patch, select_patch], dim=0)
            scores = torch.cat([scores, select_score], dim=0)
            action_id = torch.cat([action_id, select_action_id], dim=0)
                
        # data preparing
        feat_whole = feat_whole.cuda()
        # feat_patch = feat_patch.cuda()
        scores = scores.float().cuda()
        # print(action_id)
        action_id = action_id.type(torch.int64).cuda()
        
        # assert 1==2
        feat = lstm(feat_whole)
        feat_pre = lstm_pre(feat_whole)
        feat_pre = feat_pre.detach()

        pred_score = score_rgs(feat)
        
        # augmentation of selected previous data
        if combined_exemplar is not None:
            aug_helper_whole, aug_helper_score, aug_helper_action_id, aug_helper_select_id = utils.random_select_exemplar(combined_exemplar, args.num_helpers, count)
            aug_helper_whole = aug_helper_whole.cuda()
            aug_helper_action_id = aug_helper_action_id.type(torch.int64).cuda()
            # print('aug_helper_whole.shape : ',aug_helper_whole.shape ) # B=7
            aug_helper_feat = lstm_pre(aug_helper_whole)
            # if args.save_graph:
            #     aug_helper_feat = aug_helper_feat.transpose(0,1).gather(1, aug_helper_action_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,512)).squeeze(1)     # [B, 512]
            aug_helper_feat = aug_helper_feat.detach()      # [num_helpers, D]
            
            aug_feat, aug_score = utils.feat_score_aug(feat_pre[int(feat_pre.shape[0]/2) :].cpu(), aug_helper_feat.cpu(), select_score, aug_helper_score, aug_scale=args.aug_scale)

            aug_feat = aug_feat.cuda()
            aug_score = aug_score.cuda()
            score_diff = scores[int(scores.shape[0]/2) :] - aug_score
            combined_feature = torch.cat((feat[int(feat.shape[0]/2) :], aug_feat), dim=-1)
            # print('concat feat: ', combined_feature.shape )   # [B, D]
            # print('score_diff: ', score_diff.shape  )         # [B]
            if args.diff_loss:
                pred_diff = diff_rgs(combined_feature)
            elif args.aug_rgs:
                pred_aug_score = score_rgs(aug_feat)
            # print('pred_diff: ', pred_diff.shape)

        loss = 0.0
        distill_loss = 0.0
        st_pod_loss = 0.0
        diff_loss = 0.0
        ewc_loss = 0.0
        graph_distill_loss = 0.0
        mse_loss = loss_fn.mse_(pred_score, scores, mse) # mse loss
        if combined_exemplar is not None and args.diff_loss:
            diff_loss = loss_fn.mse_(pred_diff, score_diff, mse)
        if combined_exemplar is not None and args.aug_rgs:
            diff_loss = loss_fn.mse_(pred_aug_score, aug_score, mse)
        if (t!=0):
            # if args.save_graph:
            #     distill_loss = loss_fn.distill_save_graph_(feat_distill, feat_pre_distill, mse, action_id, seen_tasks) # distillation loss
            # else:
            distill_loss = loss_fn.distill_(feat, feat_pre, mse, softmax)
            # assert 1==2
        if (t!=0) and (args.approach == 'ewc'):
            ewc_loss = loss_fn.ewc_loss_(fisher, lstm, older_params, lamb=args.lambda_ewc)
        # if (t!=0) and (args.pod_loss):
        #     st_pod_loss = loss_fn.st_pod_(featmap_list, featmap_pre_list, temporal_pool=True, norm=True) # st_pod loss
        # if (t!=0) and args.graph_distill:
        #     graph_distill_loss = loss_fn.ge_graph_distill_(model_=lstm, model_pre_=lstm_pre, seen_tasks=seen_tasks, mse=mse)
        # loss = args.lambda_distill * distill_loss + args.lambda_pod * st_pod_loss + mse_loss + args.lambda_diff* diff_loss + args.lambda_graph_distill * graph_distill_loss
        loss = args.lambda_distill * distill_loss + mse_loss + args.lambda_diff* diff_loss + ewc_loss
        # loss, detail_loss = compute_loss(t, feat, feat_pre, pred_score, label, mse, ce, softmax, args)

        # print(lstm.module.spatial_mat1[1].weight[1])
        # print(lstm.module.spatial_mat1[5].weight[5])
        # connectivity_graph1 = torch.tensor(lstm.module.a).float()
        # spatial_graph1 = torch.abs(lstm.module.spatial_mat1[5].weight.cpu() * connectivity_graph1)
        # loss = loss.cuda()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse_loss.item()
        
        if (t!=0) and (args.approach!='finetune'):
            total_distill += distill_loss.item()
            # print(distill_loss.item())
        # if (t!=0) and (args.approach=='podnet'):
        #     total_st_pod += st_pod_loss.item()
        # if (t!=0) and (args.graph_distill) and (args.lambda_graph_distill != 0):
        #     total_graph_distill += graph_distill_loss.item()
        if (t!=0) and (args.approach == 'ewc'):
            total_ewc += ewc_loss.item()
        if combined_exemplar is not None and (args.diff_loss):
            total_diff += diff_loss.item()
        # if (t!=0):
            
        count += 1
        # if args.exp_name == 'debug':
        #     utils.progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | mse: %.3f | distill: %.3f | diff: %.3f | graph_distill: %.3f'% (total_loss/(batch_idx+1), total_mse/(batch_idx+1), total_distill/(batch_idx+1), total_diff/(batch_idx+1), total_graph_distill/(batch_idx+1)))
    if args.exp_name != 'debug':
        l = len(dataloader)
        print('Train process: Loss: %.3f | mse: %.3f | distill: %.3f | diff: %.3f | ewc: %.3f'% (total_loss/l, total_mse/l, total_distill/l, total_diff/l, total_ewc/l))
        # writer.add_scalar('Train_mse:', total_mse/l, epoch)
        # writer.add_scalar('Train_distill:', total_distill/l, epoch)
        # writer.add_scalar('Train_diff:', total_diff/l, epoch)
        # writer.add_scalar('Train_graph:', total_graph_distill/l, epoch)
    return lstm, score_rgs, diff_rgs, optimizer


def test_net(t, lstm, score_rgs, dataloaders, rho_matrix, rl2_matrix, args, seen_tasks=[]):
    lstm.eval()
    score_rgs.eval()
    torch.set_grad_enabled(False)
    
    # rho_matrix=np.zeros((len(dataloaders),len(dataloaders)),dtype=np.float32)
    # rl2_matrix=np.zeros((len(dataloaders),len(dataloaders)),dtype=np.float32)
    for i, a_task in enumerate(seen_tasks):
        dataloader = dataloaders[a_task]
        true_scores = []
        pred_scores = []
        for batch_idx, (feat_whole, scores, action_id) in enumerate(dataloader):
            # print('step {}'.format(step))
            feat_whole = feat_whole.cuda()
            scores = scores.float().cuda()
            action_id = action_id.type(torch.int64).cuda()
            batch_size = action_id.shape[0]

            feat = lstm(feat_whole)
            # if args.save_graph:
            #     feat = feat.transpose(0,1).gather(1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,512)).squeeze(1) 
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


def eval_net(lstm, score_rgs, dataloader, best_lstm, best_rgs, rho_best, epoch_best, L2_min, RL2_min, args, epoch=0, step='', seen_tasks=[], rho_bank=[], local_best_lstm= None, local_best_rgs=None, local_best_result=None):
    lstm.eval()
    score_rgs.eval()
    torch.set_grad_enabled(False)

    print(' {}: '.format(step), end='')
    true_scores = []
    pred_scores = []
    for batch_idx, (feat_whole, scores, action_id) in enumerate(dataloader):
        batch_size = action_id.shape[0]

        feat_whole = feat_whole.cuda()
        scores = scores.float().cuda()
        action_id = action_id.type(torch.int64).cuda()

        feat = lstm(feat_whole)
        # if args.save_graph:
        #     feat = feat.transpose(0,1).gather(1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,512)).squeeze(1) 
        pred_score = score_rgs(feat)

        pred_scores.extend(pred_score.detach().data.cpu().reshape(-1).numpy())
        true_scores.extend(scores.detach().data.cpu().reshape(-1).numpy())

    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)
    rho, p = stats.spearmanr(pred_scores, true_scores)
    L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
    RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
        true_scores.shape[0]
    print(' correlation: %.6f, L2: %.6f, RL2: %.6f' % (rho, L2, RL2), end='   ')
    

    if step=='Test':
        if len(rho_bank) < 10:
            rho_bank.append(rho)
        else:
            del rho_bank[0]
            rho_bank.append(rho)
        if rho == max(rho_bank):
            local_best_lstm = utils.get_model(lstm)
            local_best_rgs = utils.get_model(score_rgs)
            local_best_result = (epoch, rho, L2, RL2)
        local_avg_rho = sum(rho_bank) / len(rho_bank)
        if local_avg_rho >= rho_best:
            # print('* {} > {}'.format(rho, rho_best))
            rho_best = local_best_result[1]
            epoch_best = local_best_result[0]
            L2_min = local_best_result[2]
            RL2_min = local_best_result[3]
            # best_lstm = utils.get_model(lstm)
            # best_rgs = utils.get_model(score_rgs)
            best_lstm = local_best_lstm
            best_rgs = local_best_rgs
            print('*')
        print()
        print('Current best————Corr: %.6f , L2: %.6f , RL2: %.6f @ epoch %d \n' % (rho_best, L2_min, RL2_min, epoch_best))
    elif step=='Train':
        print()
    elif step=='Test_stage2':
        print()
        return rho, L2, RL2
    else:
        print('Wrong step name')
        return None

    return best_lstm, best_rgs, rho_best, epoch_best, L2_min, RL2_min, rho_bank, local_best_lstm, local_best_rgs, local_best_result

def run_exp(args, task_list, action2task, classes_name, exemplar_set):
    
    # get models
    lstm, score_rgs, diff_rgs = builder.build_moodel(args)
    lstm_pre, score_rgs_pre, _ = builder.build_moodel(args)

    mse = nn.MSELoss()
    kl = nn.KLDivLoss()

    # cuda
    lstm = lstm.cuda()
    lstm_pre = lstm_pre.cuda()
    score_rgs = score_rgs.cuda()
    diff_rgs = diff_rgs.cuda()
    mse = mse.cuda()
    kl = kl.cuda()
    softmax = nn.Softmax().cuda()
    ce = nn.CrossEntropyLoss().cuda()
    # DP
    lstm = nn.DataParallel(lstm)
    lstm_pre = nn.DataParallel(lstm_pre)
    score_rgs = nn.DataParallel(score_rgs)
    diff_rgs = nn.DataParallel(diff_rgs)

    best_lstm = utils.get_model(lstm)
    best_rgs = utils.get_model(score_rgs)
    init_diff_rgs = utils.get_model(diff_rgs)

    global fisher
    fisher = {n: torch.zeros(p.shape).cuda() for n, p in lstm.named_parameters() if p.requires_grad}


    # load data
    loaders_test = []
    start_time = time.time()
    for i in range(len(task_list)):
        _, loader_test, _ = builder.load_data(data_root=args.data_root, set_id=i, args=args, exemplar_set=None, num_tasks=args.num_tasks)
        loaders_test.append(loader_test)
    print('dataset making time cost: ', time.time()-start_time)
    
    rho_matrix=np.zeros((len(loaders_test),len(loaders_test)),dtype=np.float32)
    rl2_matrix=np.zeros((len(loaders_test),len(loaders_test)),dtype=np.float32)
    seen_tasks = []
    for t, task in enumerate(task_list):
        if not args.upper_bound:
            print('*'*60)
            print('Task {:2d} ({:s})'.format(t, classes_name[task]))
            print('*'*60)
            seen_tasks.append(task)
        else:
            print('*'*60)
            print('Running Upper-Bound')
            print('*'*60)
            seen_tasks = [0,1]
            if t >= 1:
                break

        # create writer
        global writer
        writer = SummaryWriter(os.path.join(args.ckpt_root, args.exp_name, 'tensorboard'), comment='_task_{}_{}'.format(t, classes_name[task]))
        # SummaryWriter(os.path.join(args.ckpt_root, args.exp_name, 'multi-action-tensorboard'))
        utils.freeze_model(lstm_pre)
        utils.freeze_model(lstm)

        # if args.g_e_graph:
        #     init_e_graph(lstm ,t, seen_tasks=seen_tasks)

        utils.activate_model(lstm)
        utils.activate_model(score_rgs)
        # if 'debug' in args.exp_name:
        #     for name, params in lstm.named_parameters():
        #         print(name)
        #     assert 1==2
        
        if args.optim_mode == 'new_optim':
            print('use new_optim')
            optimizer = get_optim.get_optim(lstm, score_rgs, diff_rgs, args, optim_id=args.optim_id)
        else:
            optimizer = optim.SGD([
                {'params': filter(lambda p: p.requires_grad, lstm.parameters())},
                {'params': score_rgs.parameters()},
                {'params': diff_rgs.parameters()}
            ], lr=args.base_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)
        # assert 1==2
        loader_train, loader_test, purely_task_data = builder.load_data(data_root=args.data_root, set_id=task, args=args, exemplar_set=exemplar_set, multi_actions=args.upper_bound, num_tasks=args.num_tasks)

        # run_net
        epoch_best = 0
        rho_best = 0
        L2_min = 1000
        RL2_min = 1000
        
        # 定义一个list，存邻近的10个epoch表现
        rho_bank = []
        local_best_lstm, local_best_rgs, local_best_result = None, None, None
        if not args.dataset_mixup:
            combined_exemplar = utils.reconstruct_exemplar_set(exemplar_set)
        else:
            combined_exemplar = None
        for epoch in range(0, args.num_epochs):
            print('Epoch: {}'.format(epoch))
            lstm, score_rgs, diff_rgs, optimizer = train_epoch(t, task, epoch, lstm, lstm_pre, score_rgs, diff_rgs, loader_train, optimizer, mse, kl, ce, softmax, args, seen_tasks, combined_exemplar)
            best_lstm, best_rgs, _, _, _, _, _, _, _, _ = eval_net(lstm, score_rgs, loader_train, best_lstm, best_rgs, rho_best, epoch_best, L2_min, RL2_min, args, epoch, 'Train', seen_tasks, None, None, None, None)
            best_lstm, best_rgs, rho_best, epoch_best, L2_min, RL2_min, rho_bank, local_best_lstm, local_best_rgs, local_best_result = eval_net(lstm, score_rgs, loader_test, best_lstm, best_rgs, rho_best, epoch_best, L2_min, RL2_min, args, epoch, 'Test', seen_tasks, rho_bank, local_best_lstm, local_best_rgs, local_best_result)
            writer.add_scalar('local_avg_rho:', sum(rho_bank)/len(rho_bank), epoch)
            if args.exp_name == 'debug':
                print('rho bank: ', rho_bank)
                print('local best result: ', local_best_result)
            if args.lr_decay:
                scheduler.step()
            print()

        
        utils.set_model_(lstm, best_lstm)
        utils.set_model_(lstm_pre, best_lstm)
        utils.set_model_(score_rgs, best_rgs)
        # utils.set_model_(diff_rgs, init_diff_rgs)   
        if args.approach == 'ewc':
            global older_params
            older_params = {n: p.clone().detach() for n, p in lstm.named_parameters() if p.requires_grad}
            curr_fisher = utils.compute_fisher_matrix_diag(loader_train, lstm, score_rgs,  optimizer, mse , args, seen_tasks)
            # merge fisher information, we do not want to keep fisher information for each task in memory
            for n in fisher.keys():
                # Added option to accumulate fisher over time with a pre-fixed growing alpha
                fisher[n] = (0.5 * fisher[n] + (1 - 0.5) * curr_fisher[n])
        
        # 更新exemplar set
        if args.replay:
            packed_list = []
            for data_idx in range(len(purely_task_data[2])):
                packed_list.append([purely_task_data[0][data_idx], purely_task_data[1][data_idx], purely_task_data[2][data_idx]])
            exemplar_set = utils.after_train(t, task, exemplar_set, packed_list, args, lstm, score_rgs, seen_tasks)
            # print(len(exemplar_set))
            # print(len(exemplar_set[0]))
            # assert 1==2
            if args.exp_name == 'debug':
                with open(os.path.join('./ckpt', 'debug', 'exemplar_set.txt'), 'a') as f:
                    str_w = ''
                    for i in range(len(exemplar_set)):
                        str_w += (str(len(exemplar_set[i])) + ' ')
                    f.writelines(str_w + '\n')

        # 存ckpt
        if args.save_ckpt:
            ckpt_path = './ckpt/{}/'.format(args.exp_name)  + str(t) + '_' + classes_name[task] + '_best@{}.pth'.format(epoch_best)
            # utils.save_model(best_lstm, best_rgs, epoch_best, rho_best, L2_min, RL2_min, ckpt_path)

        # 测试
        rho_matrix, rl2_matrix = test_net(t, lstm, score_rgs, loaders_test, rho_matrix, rl2_matrix, args, seen_tasks)
        np.savetxt(os.path.join('./ckpt', args.exp_name, 'rho.txt'), rho_matrix,'%.4f')
        np.savetxt(os.path.join('./ckpt', args.exp_name, 'rl2.txt'),rl2_matrix,'%.4f')


def main():
    # get exp settings
    args, task_list, action2task, classes_name = builder.build_exp()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # 保证可以复现的设定
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)  
    # torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.enabled = False

    print('classes names: ', classes_name)
    # gpus = args.gpu.split(',')
    # for gpu in gpus:
    #     occupy_gpu_memory(gpu)
    exemplar_set = [[] for _ in range(args.num_tasks)]
    # run experiment
    run_exp(args, task_list, action2task, classes_name, exemplar_set)   


    
    
if __name__ == '__main__':
    start = time.time()
    global writer
    writer = None
    main()
    print('time cost: ', time.time()-start)
