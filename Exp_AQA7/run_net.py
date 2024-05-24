from builtins import print
import sys,time
import os
import random

import sys
import numpy as np
import get_optim
import torch
import torch.nn as nn
import torch.optim as optim

from models.JRG_ASS import run_jrg
from scipy import stats
import builder
import utils
import loss_fn

from models.JRG_ASS import init_e_graph


def get_size(obj, seen=None):
    """calculate the size of an object"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)

    if isinstance(obj, (list, tuple, dict, set)):
        size += sum(get_size(item, seen) for item in obj)
    elif hasattr(obj, '__dict__'):

        size += get_size(obj.__dict__, seen)
    return size

def train_epoch(t, task, epoch, jrg, jrg_pre, score_rgs, diff_rgs, dataloader, optimizer, mse, kl, ce, softmax, args, seen_tasks=[], combined_exemplar=None):
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




    count = 0
    for batch_idx, (feat_whole, feat_patch, scores, action_id) in enumerate(dataloader):
        batch_size = action_id.shape[0]
        # old data
        if combined_exemplar is not None:
            # assert 1==0
            select_whole, select_patch, select_score, select_action_id, select_id = utils.random_select_exemplar(combined_exemplar, batch_size, count)
            feat_whole = torch.cat([feat_whole, select_whole], dim=0)
            feat_patch = torch.cat([feat_patch, select_patch], dim=0)
            scores = torch.cat([scores, select_score], dim=0)
            action_id = torch.cat([action_id, select_action_id], dim=0)
                
        # data preparing
        feat_whole = feat_whole.cuda()
        feat_patch = feat_patch.cuda()
        scores = scores.float().cuda()
        # print(action_id)
        action_id = action_id.type(torch.int64).cuda()
        
        # assert 1==2

        feat, featmap_list= run_jrg(jrg, feat_whole, feat_patch, save_graph=args.save_graph, seen_tasks=seen_tasks, args=args)
        feat_pre, featmap_pre_list = run_jrg(jrg_pre, feat_whole, feat_patch, save_graph=args.save_graph, seen_tasks=seen_tasks, args=args)
        feat_pre = feat_pre.detach()

        if args.save_graph:
            feat_distill = feat
            feat_pre_distill = feat_pre
            feat = feat.transpose(0,1).gather(1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,512)).squeeze(1)      # [B, 512]
            feat_pre = feat_pre.transpose(0,1).gather(1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,512)).squeeze(1) 



        pred_score = score_rgs(feat)
        
        # augmentation of selected previous data
        if combined_exemplar is not None:
            aug_helper_whole, aug_helper_patch, aug_helper_score, aug_helper_action_id, aug_helper_select_id = utils.random_select_exemplar(combined_exemplar, args.num_helpers, count)
            aug_helper_whole = aug_helper_whole.cuda()
            aug_helper_patch = aug_helper_patch.cuda()
            aug_helper_action_id = aug_helper_action_id.type(torch.int64).cuda()
            aug_helper_feat, _ = run_jrg(jrg_pre, aug_helper_whole, aug_helper_patch, save_graph=args.save_graph, seen_tasks=seen_tasks, args=args)
            if args.save_graph:
                aug_helper_feat = aug_helper_feat.transpose(0,1).gather(1, aug_helper_action_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,512)).squeeze(1)     # [B, 512]
            aug_helper_feat = aug_helper_feat.detach()      # [num_helpers, D]
            
            aug_feat, aug_score = utils.feat_score_aug(feat_pre[int(feat_pre.shape[0]/2) :].cpu(), aug_helper_feat.cpu(), select_score, aug_helper_score, aug_scale=args.aug_scale, with_weight=args.aug_w_weight)

            aug_feat = aug_feat.cuda()
            aug_score = aug_score.cuda()
            score_diff = scores[int(scores.shape[0]/2) :] - aug_score
            combined_feature = torch.cat((feat[int(feat.shape[0]/2) :], aug_feat), dim=-1)
            if args.diff_loss:
                pred_diff = diff_rgs(combined_feature)
            elif args.aug_rgs:
                pred_aug_score = score_rgs(aug_feat)

        loss = 0.0
        distill_loss = 0.0
        st_pod_loss = 0.0
        diff_loss = 0.0
        graph_distill_loss = 0.0
        mse_loss = loss_fn.mse_(pred_score, scores, mse) # mse loss
        if combined_exemplar is not None and args.diff_loss:
            diff_loss = loss_fn.mse_(pred_diff, score_diff, mse)
        if combined_exemplar is not None and args.aug_rgs:
            diff_loss = loss_fn.mse_(pred_aug_score, aug_score, mse)
        if (t!=0):
            if args.save_graph:
                distill_loss = loss_fn.distill_save_graph_(feat_distill, feat_pre_distill, mse, action_id, seen_tasks) # distillation loss
            else:
                distill_loss = loss_fn.distill_(feat, feat_pre, mse, softmax)
            # assert 1==2
        if (t!=0) and (args.pod_loss):
            st_pod_loss = loss_fn.st_pod_(featmap_list, featmap_pre_list, temporal_pool=True, norm=True) # st_pod loss
        if (t!=0) and args.graph_distill:
            graph_distill_loss = loss_fn.ge_graph_distill_(model_=jrg, model_pre_=jrg_pre, seen_tasks=seen_tasks, mse=mse)
        loss = args.lambda_distill * distill_loss + args.lambda_pod * st_pod_loss + mse_loss + args.lambda_diff* diff_loss + args.lambda_graph_distill * graph_distill_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total_loss += loss.item()
        total_mse += mse_loss.item()
        
        if (t!=0) and (args.approach!='finetune'):
            total_distill += distill_loss.item()
            # print(distill_loss.item())
        if (t!=0) and (args.approach=='podnet'):
            total_st_pod += st_pod_loss.item()
        if (t!=0) and (args.graph_distill) and (args.lambda_graph_distill != 0):
            total_graph_distill += graph_distill_loss.item()
        if combined_exemplar is not None and (args.diff_loss):
            total_diff += diff_loss.item()

            
        count += 1
    return jrg, score_rgs, diff_rgs, optimizer


def test_net(t, jrg, score_rgs, dataloaders, rho_matrix, rl2_matrix, args, seen_tasks=[]):

    jrg.eval()
    score_rgs.eval()
    torch.set_grad_enabled(False)

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

            feat, _ = run_jrg(jrg, feat_whole, feat_patch, save_graph=args.save_graph, seen_tasks=seen_tasks, args=args)
            if args.save_graph:
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


def eval_net(jrg, score_rgs, dataloader, best_jrg, best_rgs, rho_best, epoch_best, L2_min, RL2_min, args, epoch=0, step='', seen_tasks=[], rho_bank=[], local_best_jrg= None, local_best_rgs=None, local_best_result=None):
    jrg.eval()
    score_rgs.eval()
    torch.set_grad_enabled(False)

    print(' {}: '.format(step), end='')
    true_scores = []
    pred_scores = []
    for batch_idx, (feat_whole, feat_patch, scores, action_id) in enumerate(dataloader):
        batch_size = action_id.shape[0]

        feat_whole = feat_whole.cuda()
        feat_patch = feat_patch.cuda()
        scores = scores.float().cuda()
        action_id = action_id.type(torch.int64).cuda()

        feat, _ = run_jrg(jrg, feat_whole, feat_patch, save_graph=args.save_graph, seen_tasks=seen_tasks, args=args)
        if args.save_graph:
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
    print(' correlation: %.6f, L2: %.6f, RL2: %.6f' % (rho, L2, RL2), end='   ')
    

    if step=='Test':
        if len(rho_bank) < 10:
            rho_bank.append(rho)
        else:
            del rho_bank[0]
            rho_bank.append(rho)
        if rho == max(rho_bank):
            local_best_jrg = utils.get_model(jrg)
            local_best_rgs = utils.get_model(score_rgs)
            local_best_result = (epoch, rho, L2, RL2)
        local_avg_rho = sum(rho_bank) / len(rho_bank)
        if local_avg_rho >= rho_best:
            # print('* {} > {}'.format(rho, rho_best))
            rho_best = local_best_result[1]
            epoch_best = local_best_result[0]
            L2_min = local_best_result[2]
            RL2_min = local_best_result[3]
            # best_jrg = utils.get_model(jrg)
            # best_rgs = utils.get_model(score_rgs)
            best_jrg = local_best_jrg
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

    return best_jrg, best_rgs, rho_best, epoch_best, L2_min, RL2_min, rho_bank, local_best_jrg, local_best_rgs, local_best_result

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

    

    # load data
    loaders_test = []
    start_time = time.time()
    for i in range(len(task_list)):
        _, _, _, loader_test, _ = builder.load_data(data_root=args.data_root, set_id=i, args=args, exemplar_set=None)
        loaders_test.append(loader_test)
    print('dataset making time cost: ', time.time()-start_time)
    
    rho_matrix=np.zeros((len(loaders_test),len(loaders_test)),dtype=np.float32)
    rl2_matrix=np.zeros((len(loaders_test),len(loaders_test)),dtype=np.float32)
    seen_tasks = []
    for t, task in enumerate(task_list):
        print('*'*60)
        print('Task {:2d} ({:s})'.format(t, classes_name[task]))
        print('*'*60)
        seen_tasks.append(task)

        # create writer
        global writer
        utils.freeze_model(jrg_pre)
        utils.freeze_model(jrg)

        if args.g_e_graph and (not args.graph_random_init):
            print('init graph')
            init_e_graph(jrg ,t, seen_tasks=seen_tasks)

        utils.activate_model(jrg)
        utils.activate_model(score_rgs)
        # if 'debug' in args.exp_name:
        #     for name, params in jrg.named_parameters():
        #         print(name)
        #     assert 1==2
        
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
        dataset_train, dataset_test, loader_train, loader_test, purely_task_data = builder.load_data(data_root=args.data_root, set_id=task, args=args, exemplar_set=exemplar_set)

        # run_net
        epoch_best = 0
        rho_best = 0
        L2_min = 1000
        RL2_min = 1000
        
        rho_bank = []
        local_best_jrg, local_best_rgs, local_best_result = None, None, None
        if not args.dataset_mixup:
            combined_exemplar = utils.reconstruct_exemplar_set(exemplar_set)
        else:
            combined_exemplar = None
        for epoch in range(0, args.num_epochs):
            print('Epoch: {}'.format(epoch))
            jrg, score_rgs, diff_rgs, optimizer = train_epoch(t, task, epoch, jrg, jrg_pre, score_rgs, diff_rgs, loader_train, optimizer, mse, kl, ce, softmax, args, seen_tasks, combined_exemplar)
            best_jrg, best_rgs, _, _, _, _, _, _, _, _ = eval_net(jrg, score_rgs, loader_train, best_jrg, best_rgs, rho_best, epoch_best, L2_min, RL2_min, args, epoch, 'Train', seen_tasks, None, None, None, None)
            best_jrg, best_rgs, rho_best, epoch_best, L2_min, RL2_min, rho_bank, local_best_jrg, local_best_rgs, local_best_result = eval_net(jrg, score_rgs, loader_test, best_jrg, best_rgs, rho_best, epoch_best, L2_min, RL2_min, args, epoch, 'Test', seen_tasks, rho_bank, local_best_jrg, local_best_rgs, local_best_result)
            # writer.add_scalar('local_avg_rho:', sum(rho_bank)/len(rho_bank), epoch)
            if args.exp_name == 'debug':
                print('rho bank: ', rho_bank)
                print('local best result: ', local_best_result)
            if args.lr_decay:
                scheduler.step()
            print()

        
        utils.set_model_(jrg, best_jrg)
        utils.set_model_(jrg_pre, best_jrg)
        utils.set_model_(score_rgs, best_rgs)
        
        # update exemplar set
        if args.replay:
            packed_list = []
            for data_idx in range(len(purely_task_data[2])):
                packed_list.append([purely_task_data[0][data_idx], purely_task_data[1][data_idx], purely_task_data[2][data_idx]])
            exemplar_set = utils.after_train(t, task, exemplar_set, packed_list, args, jrg, score_rgs, seen_tasks)
            if args.exp_name == 'debug':
                with open(os.path.join('./ckpt', 'debug', 'exemplar_set.txt'), 'a') as f:
                    str_w = ''
                    for i in range(len(exemplar_set)):
                        str_w += (str(len(exemplar_set[i])) + ' ')
                    f.writelines(str_w + '\n')

                torch.save(exemplar_set, os.path.join('./ckpt', 'debug', 'exemplar_set_torch_{}.torch'.format(t)))

        # 存ckpt
        if args.save_ckpt:
            ckpt_path = './ckpt/{}/'.format(args.exp_name)  + str(t) + '_' + classes_name[task] + '_best@{}.pth'.format(epoch_best)
            utils.save_model(best_jrg, best_rgs, epoch_best, rho_best, L2_min, RL2_min, ckpt_path)

        # 测试
        rho_matrix, rl2_matrix = test_net(t, jrg, score_rgs, loaders_test, rho_matrix, rl2_matrix, args, seen_tasks)
        np.savetxt(os.path.join('./ckpt', args.exp_name, 'rho.txt'), rho_matrix,'%.4f')
        np.savetxt(os.path.join('./ckpt', args.exp_name, 'rl2.txt'),rl2_matrix,'%.4f')


def main():
    # get exp settings
    args, task_list, action2task, classes_name = builder.build_exp()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
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

    
    exemplar_set = [[] for _ in range(6)]
    # run experiment
    run_exp(args, task_list, action2task, classes_name, exemplar_set)   


    
    
if __name__ == '__main__':
    start = time.time()
    global writer
    writer = None
    main()
    print('time cost: ', time.time()-start)
