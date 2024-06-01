import time
import sys
import os
import copy
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# self-written packages
import builder
import utils
import get_optim
import loss_fn
from models.JRG_ASS_AGSG import init_e_graph


def train_epoch(t, task, epoch, jrg, jrg_pre, score_rgs, diff_rgs, dataloader, optimizer, mse, kl, ce, softmax, args, seen_tasks=[], combined_exemplar=None, current_importance=None):
    jrg.train()
    score_rgs.train()
    diff_rgs.train()
    jrg_pre.eval()
    torch.set_grad_enabled(True)

    total_loss = 0.0
    total_mse = 0.0
    total_distill = 0.0
    total_st_pod = 0.0
    total_diff = 0.0
    total_graph_distill = 0.0

    loss_ = 0.0
    margin_loss_ = 0.0
    orth_loss_ = 0.0
    l2_loss_ = 0.0
    sabs_loss_ = 0.0
    # CL loss
    feat_distill_loss_ = 0.0
    diff_loss_ = 0.0
    pod_loss_ = 0.0

    acc_cnt = 0
    tot_cnt = 0
    gama = args.gama
    
    for batch_idx, (whole1, whole2, action_id_1, action_id_2) in enumerate(dataloader):
        # st_time = time.time()

        whole1 = whole1.cuda()
        whole2 = whole2.cuda()
        action_id_1 = action_id_2.cuda()
        action_id_2 = action_id_2.cuda()    # [B]
        

        # get old data from combined exemplar set
        if combined_exemplar is not None:
            # assert 1==0
            select_whole, select_score, select_action_id, select_id = utils.random_select_exemplar_BEST_(combined_exemplar, args.batch_size)
            select_whole = select_whole.cuda()
            select_score = select_score.cuda()
            select_action_id = select_action_id.cuda()
            # print(select_action_id)
            select_whole_prime = torch.cat((select_whole[1:], select_whole[0].unsqueeze(0)), dim=0)
            select_score_prime = torch.cat((select_score[1:], select_score[0].unsqueeze(0)), dim=0)
            select_action_id_prime = torch.cat((select_action_id[1:], select_action_id[0].unsqueeze(0)), dim=0)

            select_whole1 = torch.Tensor([]).cuda()
            select_whole2 = torch.Tensor([]).cuda()
            scores_1 = torch.Tensor([]).cuda()
            scores_2 = torch.Tensor([]).cuda()
            select_action_id_1 = torch.Tensor([]).cuda()
            select_action_id_2 = torch.Tensor([]).cuda()
            for i in range(len(select_score)):
                if select_score[i] >= select_score_prime[i]:
                    select_whole1 = torch.cat([select_whole1, select_whole[i].unsqueeze(0)])
                    select_whole2 = torch.cat([select_whole2, select_whole_prime[i].unsqueeze(0)])
                    scores_1 = torch.cat([scores_1, select_score[i].reshape(1, -1)], dim=0)
                    scores_2 = torch.cat([scores_2, select_score_prime[i].reshape(1, -1)], dim=0)
                    select_action_id_1 = torch.cat([select_action_id_1, select_action_id[i].reshape(-1)], dim=0)
                    select_action_id_2 = torch.cat([select_action_id_2, select_action_id_prime[i].reshape(-1)], dim=0)
                else:
                    select_whole1 = torch.cat([select_whole1, select_whole_prime[i].unsqueeze(0)])
                    select_whole2 = torch.cat([select_whole2, select_whole[i].unsqueeze(0)])
                    scores_1 = torch.cat([scores_1, select_score_prime[i].reshape(1, -1)], dim=0)
                    scores_2 = torch.cat([scores_2, select_score[i].reshape(1, -1)], dim=0)
                    select_action_id_1 = torch.cat([select_action_id_1, select_action_id_prime[i].reshape(-1)], dim=0)
                    select_action_id_2 = torch.cat([select_action_id_2, select_action_id[i].reshape(-1)], dim=0)
            whole1 = torch.cat([whole1, select_whole1], dim=0)
            whole2 = torch.cat([whole2, select_whole2], dim=0)
            
            action_id_1 = torch.cat([action_id_1, select_action_id_1], dim=0)
            action_id_2 = torch.cat([action_id_2, select_action_id_2], dim=0)
        # if args.exp_name == 'debug':
        #     print('t1: ', time.time()-st_time)
        action_id = torch.cat([action_id_1, action_id_2]).long()
        
        
        feat, featmap, cosine_tensor, l2_tensor, _ = jrg(torch.cat((whole1, whole2), dim=0))
        feat_pre, featmap_pre, _, _, _ = jrg_pre(torch.cat((whole1, whole2), dim=0))

        if args.save_graph:
            # print('here')
            feat_distill = feat
            feat_pre_distill = feat_pre

            feat = feat.gather(1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,512)).squeeze(1)      # [B, 512]
            feat_pre = feat_pre.gather(1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,512)).squeeze(1) 

        pred = score_rgs(feat)

        pred1 = pred[0:whole1.shape[0]]
        pred2 = pred[whole1.shape[0]::]

        if combined_exemplar is not None:
            aug_helper_whole, aug_helper_score, aug_helper_action_id, aug_helper_select_id = utils.random_select_exemplar_BEST_(combined_exemplar, args.num_helpers, count=None)
            aug_helper_whole = aug_helper_whole.cuda()
            aug_helper_score = aug_helper_score.cuda()
            aug_helper_action_id = aug_helper_action_id.type(torch.int64).cuda()

            aug_helper_feat, _, _, _, _ = jrg_pre(aug_helper_whole)
            if args.save_graph:
                aug_helper_feat = aug_helper_feat.gather(1, aug_helper_action_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,512)).squeeze(1)     # [B, 512]
            aug_helper_feat = aug_helper_feat.detach()      # [num_helpers, D]
            
            # print(0)
            feat2aug = torch.cat([feat_pre[int(feat_pre.shape[0]/2)-len(select_whole): int(feat_pre.shape[0]/2)], feat_pre[int(feat_pre.shape[0])-len(select_whole): ]], dim=0)
            score2aug = torch.cat([scores_1, scores_2]).cuda()
            aug_feat, aug_score = utils.feat_score_aug_BEST_(feat2aug, aug_helper_feat, score2aug, aug_helper_score, aug_scale=args.aug_scale)


            score_diff = score2aug - aug_score
            feat2compute_diff = torch.cat([feat[int(feat.shape[0]/2)-len(select_whole): int(feat.shape[0]/2)], feat[int(feat.shape[0])-len(select_whole): ]], dim=0)
            combined_feature = torch.cat((feat2compute_diff, aug_feat), dim=-1)

            if args.diff_loss:
                pred_diff = diff_rgs(combined_feature)
            elif args.aug_rgs:
                pred_aug_score = score_rgs(aug_feat)

        margin_tensor = (gama - pred1 + pred2)
        margin_tensor[margin_tensor < 0] = 0
        margin_loss = torch.mean(margin_tensor, dim=0)

        # feature distillation
        feat_distill_loss = torch.tensor(0.0)
        if t>0 and args.feat_distill:
            if args.save_graph:
                feat_distill_loss = args.lambda_distill * loss_fn.feature_distill_save_graph_(feat_distill, feat_pre_distill, mse, action_id, seen_tasks, args)
            else:
                feat_distill_loss = args.lambda_distill * loss_fn.feature_distill_(feat, feat_pre, mse)

        # pod loss or afc loss
        pod_loss = torch.tensor(0.0)
        if t>0 and (args.pod_loss or args.afc):
            if current_importance is None:
                old_importance = [1]

            else:

                old_importance = current_importance
            pod_loss = args.lambda_pod * args.lambda_afc * loss_fn.pod_(featmap_list=[featmap], featmap_pre_list=[featmap_pre], norm=True, args=args, old_importance=[old_importance], use_importance=args.afc)

        # ewc loss 
        if (t!=0) and args.ewc:
            ewc_loss = loss_fn.ewc_loss_(fisher, jrg, older_params, lamb=args.lambda_ewc)

        # difference loss
        diff_loss = torch.tensor(0.0)
        if t>0 and (combined_exemplar is not None) and args.diff_loss:
            diff_loss = args.lambda_diff * loss_fn.mse_(pred_diff, score_diff, mse)
        
        ## sum up all the losses
        loss = margin_loss + feat_distill_loss + diff_loss  + pod_loss #+ l2_loss
        # print(loss)
        
        # if is_train:
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        # if args.exp_name == 'debug':
        #     print('t4: ', time.time()-st_time)

        loss_+= loss.item()
        margin_loss_ += margin_loss.item()
        feat_distill_loss_ += feat_distill_loss.item()
        diff_loss_ += diff_loss.item()
        pod_loss_ += pod_loss.item()
        # orth_loss_ += orth_loss.item()
        # l2_loss_ += l2_loss.item()
        # sabs_loss_ += sabs_loss.item()

        # acc computation
        acc_cnt += torch.sum((pred1 > pred2).int()).detach().cpu().numpy()
        tot_cnt += margin_tensor.shape[0]

    L =  len(dataloader)
    acc_rate = acc_cnt / tot_cnt
    avg_loss = loss_ / L
    avg_margin = margin_loss_ / L
    avg_feat_distill = feat_distill_loss_ / L
    avg_diff = diff_loss_ / L
    avg_pod = pod_loss_ / L
    writer.add_scalar('Task {} margin loss'.format(t), avg_margin, epoch)
    print('Train process: Loss: %.3f | F_distill: %.3f | Diff: %.3f | POD: %.3f | acc: %.3f' % (avg_loss, avg_feat_distill, avg_diff, avg_pod, acc_rate))

    return jrg, score_rgs, diff_rgs, optimizer

def test_net(t, jrg, score_rgs, dataloaders, acc_matrix, args, seen_tasks=[]):

    jrg.eval()
    score_rgs.eval()
    torch.set_grad_enabled(False)
    gama = args.gama
    for i, a_task in enumerate(seen_tasks):
        acc_cnt = 0
        tot_cnt = 0
        dataloader = dataloaders[a_task]
        for batch_idx, (whole1, whole2, action_id_1, action_id_2) in enumerate(dataloader):
            whole1 = whole1.cuda()
            whole2 = whole2.cuda()
            action_id_1 = action_id_2.cuda()
            action_id_2 = action_id_2.cuda()    # [B]
            action_id = torch.cat([action_id_1, action_id_2])

            feat, _, _, _, _ = jrg(torch.cat((whole1, whole2), dim=0))
            if args.save_graph:
                feat = feat.gather(1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,512)).squeeze(1) 
            pred = score_rgs(feat)
            pred1 = pred[0:whole1.shape[0]]
            pred2 = pred[whole1.shape[0]::]
            mean_pred = torch.mean(pred)
            bias_ = 1e-6
            # normalized_pred1 = (pred1 - mean_pred) / torch.sqrt(torch.sum((pred1-mean_pred)**2)+bias_)
            # normalized_pred2 = (pred2 - mean_pred) / torch.sqrt(torch.sum((pred2-mean_pred)**2)+bias_)
            margin_tensor = (gama - pred1 + pred2)
            margin_tensor[margin_tensor < 0] = 0
            margin_loss = torch.mean(margin_tensor, dim=0)

            loss = margin_loss #+ l2_loss
            acc_cnt += torch.sum((pred1 > pred2).int()).detach().cpu().numpy()
            tot_cnt += margin_tensor.shape[0]

        
        accuracy = acc_cnt / tot_cnt
        acc_matrix[i][t] = accuracy
    return acc_matrix

def eval_net(t, jrg, score_rgs, dataloader, best_jrg, best_rgs, acc_best, epoch_best, args, epoch=0, step='', seen_tasks=[], rho_bank=[]):
    jrg.eval()
    score_rgs.eval()
    torch.set_grad_enabled(False)
    gama = args.gama
    print(' {}: '.format(step), end='')
    acc_cnt = 0
    tot_cnt = 0
    for batch_idx, (whole1, whole2, action_id_1, action_id_2) in enumerate(dataloader):
        whole1 = whole1.cuda()
        whole2 = whole2.cuda()
        action_id_1 = action_id_2.cuda()
        action_id_2 = action_id_2.cuda()    # [B]
        action_id = torch.cat([action_id_1, action_id_2])
        # print(torch.cat((whole1, whole2), dim=0).shape)
        feat, _,  _, _, _ = jrg(torch.cat((whole1, whole2), dim=0))
        if args.save_graph:
            feat = feat.gather(1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,512)).squeeze(1) 
        pred = score_rgs(feat)
        pred1 = pred[0:whole1.shape[0]]
        pred2 = pred[whole1.shape[0]::]
        mean_pred = torch.mean(pred)
        bias_ = 1e-6
        # normalized_pred1 = (pred1 - mean_pred) / torch.sqrt(torch.sum((pred1-mean_pred)**2)+bias_)
        # normalized_pred2 = (pred2 - mean_pred) / torch.sqrt(torch.sum((pred2-mean_pred)**2)+bias_)
        margin_tensor = (gama - pred1 + pred2)
        margin_tensor[margin_tensor < 0] = 0

        acc_cnt += torch.sum((pred1 > pred2).int()).detach().cpu().numpy()
        tot_cnt += margin_tensor.shape[0]

    
    accuracy = acc_cnt / tot_cnt
    writer.add_scalar('Task_{}_{}_acc'.format(t, step), accuracy, epoch)
    print('  acc: {}'.format(accuracy))
    if step == 'Test':
        if accuracy >= acc_best:
            local_jrg = utils.get_model(jrg)
            local_rgs = utils.get_model(score_rgs)
            acc_best = accuracy
            epoch_best = epoch
            best_jrg = local_jrg
            best_rgs = local_rgs
        print('Current best: acc: {} @ eopch {}'.format(acc_best, epoch_best))
    # else:
    #     print()
    return best_jrg, best_rgs, acc_best, epoch_best

def run_exp(args, task_list, action2task, classes_name, exemplar_set):
    # get models
    jrg, score_rgs, diff_rgs = builder.build_moodel(args)
    jrg_pre, score_rgs_pre, _ = builder.build_moodel(args)

    mse = nn.MSELoss()
    kl = nn.KLDivLoss()

    # cuda
    jrg = jrg.cuda()
    jrg_pre = jrg_pre.cuda()
    score_rgs = score_rgs.cuda()
    diff_rgs = diff_rgs.cuda()
    mse = mse.cuda()
    kl = kl.cuda()
    softmax = nn.Softmax().cuda()
    ce = nn.CrossEntropyLoss().cuda()
    # DP
    jrg = nn.DataParallel(jrg)
    jrg_pre = nn.DataParallel(jrg_pre)
    score_rgs = nn.DataParallel(score_rgs)
    diff_rgs = nn.DataParallel(diff_rgs)

    best_jrg = utils.get_model(jrg)
    best_rgs = utils.get_model(score_rgs)
    init_diff_rgs = utils.get_model(diff_rgs)

    global fisher
    fisher = {n: torch.zeros(p.shape).cuda() for n, p in jrg.named_parameters() if p.requires_grad}

    # load data
    loaders_test = []
    start_time = time.time()
    for i in range(len(task_list)):
        _, loader_test, _, _ = builder.load_data(split_root=args.split_root, feature_root=args.feature_root, set_id=i, batch_size=args.batch_size, exemplar_set=None, args=args)
        loaders_test.append(loader_test)
    print('dataset making time cost: ', time.time()-start_time)
    
    acc_matrix=np.zeros((len(loaders_test),len(loaders_test)),dtype=np.float32)
    rl2_matrix=np.zeros((len(loaders_test),len(loaders_test)),dtype=np.float32)
    seen_tasks = []
    current_importance = None
    for t, task in enumerate(task_list):
        print('*'*60)
        print('Task {:2d} ({:s})'.format(t, classes_name[task]))
        print('*'*60)
        seen_tasks.append(task)

        # create writer
        global writer
        writer = SummaryWriter(os.path.join(args.ckpt_root, args.exp_name, 'tensorboard'), comment='_task_{}_{}'.format(t, classes_name[task]))
        # SummaryWriter(os.path.join(args.ckpt_root, args.exp_name, 'multi-action-tensorboard'))
        utils.freeze_model(jrg_pre)
        utils.freeze_model(jrg)

        if args.g_e_graph:
            print('init graph')
            init_e_graph(jrg ,t, seen_tasks=seen_tasks)

        utils.activate_model(jrg)
        utils.activate_model(score_rgs)
        
        if args.optim_mode == 'new_optim':
            print('use new_optim')
            optimizer = get_optim.get_optim(jrg, score_rgs, diff_rgs, args, optim_id=args.optim_id)
        else:
            optimizer = optim.SGD([
                {'params': filter(lambda p: p.requires_grad, jrg.parameters()), 'lr': args.base_lr * args.lr_factor},
                {'params': score_rgs.parameters()},
                {'params': diff_rgs.parameters()}
            ], lr=args.base_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)
        # assert 1==2
        loader_train, loader_test, purely_task_data, _ = builder.load_data(split_root=args.split_root, feature_root=args.feature_root, set_id=task, batch_size=args.batch_size, exemplar_set=exemplar_set, args=args)

        # run_net
        # 训练当前任务
        epoch_best = 0
        acc_best = 0
        L2_min = 1000
        RL2_min = 1000
        
        rho_bank = []
        local_best_jrg, local_best_rgs, local_best_result = None, None, None
        if not args.dataset_mixup:
            combined_exemplar = utils.reconstruct_exemplar_set_BEST_(exemplar_set)
        else:
            combined_exemplar = None
        
        for epoch in range(0, args.num_epochs):
            print('Epoch: {}'.format(epoch))
            jrg, score_rgs, diff_rgs, optimizer = train_epoch(t, task, epoch, jrg, jrg_pre, score_rgs, diff_rgs, loader_train, optimizer, mse, kl, ce, softmax, args, seen_tasks, combined_exemplar, current_importance=current_importance)
            # if epoch % 1 == 0:
            best_jrg, best_rgs, _, _ = eval_net(t, jrg, score_rgs, loader_train, best_jrg, best_rgs, acc_best, epoch_best, args, epoch, 'Train', seen_tasks)
            best_jrg, best_rgs, acc_best, epoch_best = eval_net(t, jrg, score_rgs, loader_test, best_jrg, best_rgs, acc_best, epoch_best, args, epoch, 'Test', seen_tasks)
            if args.lr_decay:
                scheduler.step()
            print()

        utils.set_model_(jrg, best_jrg)
        utils.set_model_(jrg_pre, best_jrg)
        utils.set_model_(score_rgs, best_rgs)

        if args.afc:
            current_importance = utils.my_update_importance(jrg, jrg_pre, score_rgs, diff_rgs, loader_train, optimizer, mse, args, seen_tasks)
        
        
        # utils.set_model_(diff_rgs, init_diff_rgs)   # 每次初始化一个新的difference regressor
        if args.ewc:
            global older_params
            older_params = {n: p.clone().detach() for n, p in jrg.named_parameters() if p.requires_grad}
            curr_fisher = utils.compute_fisher_matrix_diag(loader_train, jrg, score_rgs,  optimizer, mse , args, seen_tasks)
            # merge fisher information, we do not want to keep fisher information for each task in memory
            for n in fisher.keys():
                # Added option to accumulate fisher over time with a pre-fixed growing alpha
                fisher[n] = (0.5 * fisher[n] + (1 - 0.5) * curr_fisher[n])


        # 更新exemplar set
        if args.replay:
            packed_list = []
            # forward training data of current tasks to obtain the features
            exemplar_loader = builder.build_loader_from_list_(task, purely_task_data, args=args)    # build the loader of training data
            exemplar_feat, exemplar_score, exemplar_id = utils.get_feature_pseudo_score_(jrg, score_rgs, exemplar_loader, seen_tasks, args)   # forward data and obtain the features
            for i in range(len(exemplar_feat)):    # Pack up the features, scores and ids
                packed_list.append([exemplar_feat[i], exemplar_score[i], exemplar_id[i]])
            # reconstruct the new exemplar set
            exemplar_set = utils.after_train_BEST_(t, task, exemplar_set, packed_list, purely_task_data, args, seen_tasks)
            # # constrcut the pair-wise ranking tasks by the exemplar set
            # pass
            if args.exp_name == 'debug':
                print('recording size of exemplar set')
                with open(os.path.join('./ckpt', 'debug', 'exemplar_set.txt'), 'a') as f:
                    str_w = ''
                    for i in range(len(exemplar_set)):
                        str_w += (str(len(exemplar_set[i])) + ' ')
                    f.writelines(str_w + '\n')

        # 存ckpt
        if args.save_ckpt:
            ckpt_path = './ckpt/{}/'.format(args.exp_name)  + str(t) + '_' + classes_name[task] + '_best@{}.pth'.format(epoch_best)
            utils.save_model(best_jrg, best_rgs, epoch_best, acc_best, L2_min, RL2_min, ckpt_path)

        # 测试
        acc_matrix = test_net(t, jrg, score_rgs, loaders_test, acc_matrix, args, seen_tasks)
        np.savetxt(os.path.join('./ckpt', args.exp_name, 'rho.txt'), acc_matrix,'%.4f')
        # np.savetxt(os.path.join('./ckpt', args.exp_name, 'rl2.txt'),rl2_matrix,'%.4f')

    return 

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

    exemplar_set = [[] for _ in range(args.num_tasks)]
    # run experiment
    run_exp(args, task_list, action2task, classes_name, exemplar_set)  

if __name__ == '__main__':
    start = time.time()
    global writer
    writer = None
    main()
    print('time cost: ', time.time()-start)