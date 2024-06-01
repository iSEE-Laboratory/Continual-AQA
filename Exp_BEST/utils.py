import os,sys
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm
import time
import scipy
import builder
import copy
# from models.JRG_ASS import run_jrg

# 该部分代码是在训练和测试过程中会使用的代码
# ======================================================== 
def get_model(model):
    # 返回的是state_dict
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def activate_model(model):
    for param in model.parameters():
        param.requires_grad = True
    return

def save_model(jrg_state_dict, rgs_state_dict, epoch_best, rho_best, L2, RL2, file_path):
    torch.save({
                'jrg' : jrg_state_dict,
                'regressor' : rgs_state_dict,
                'epoch_best': epoch_best,
                'rho_best' : rho_best,
                'L2_min' : L2,
                'RL2_min' : RL2,
                }, file_path)
    return
# ======================================================== 

def fisher_z(score_list):
    score_list = np.array(score_list)
    z_transform = 0.5 * np.log((1+score_list)/(1-score_list))
    mean_z = np.mean(z_transform)
    final_score = (np.e**(2*mean_z)-1) / (np.e**(2*mean_z)+1)
    return final_score


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


# ================= 输出格式  =======================
TOTAL_BAR_LENGTH = 8.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    # L.append('  Step: %s' % format_time(step_time))
    L.append('  Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    #for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
    #    sys.stdout.write(' ')

    # Go back to the center of the bar.
    #for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
    #    sys.stdout.write('\b')
    sys.stdout.write(' %d/%d   ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

# ================= 特征-分数增强算法 =======================
def feat_score_aug(feature1, feature_list, score1, score_list, aug_scale=0.3, count=None):
    """
        b: batch size
        d: feature dimension
        n: number of helpers
        feature1: [b, d]
        feature_list: [n, d]
        score1: [b]
        score_list: [n]
    """
    # print(feature1.shape ) # [B, 512]
    # for feature2 in feature_list:
    #     print(feature2.shape)
    #     break
    # assert 1==2
    feature_diff = [(feature2 - feature1) for feature2 in feature_list]
    score_diff = [(score2 - score1) for score2 in score_list]

    aug_vector = torch.zeros(feature1.shape)
    aug_score = torch.zeros(score1.shape)
    for diff in feature_diff:
        aug_vector += diff
    for diff in score_diff:
        aug_score += diff

    if count is not None:
        np.random.seed(count)
    r = np.random.normal(loc=0.0, scale=aug_scale)         
    # print('r:', r)
    aug_feat = feature1 + (r/len(score_diff))* aug_vector
    aug_score = score1 + (r/len(score_diff)) * aug_score
    return aug_feat, aug_score
# ========================================================

# ==================== after train =======================
def after_train(t, task, exemplar_set, packed_list, args, jrg, score_rgs, seen_tasks):
    m = args.memory_size / (t+1)
    packed_sorted_list = sorted(packed_list,key=lambda x:x[2])
    if t != 0:
        m_p = int(args.memory_size / t)

        reduce_exemplar_sets(m, m_p, exemplar_set)
    data2save = get_m_exemplar_simple(task, packed_sorted_list=packed_sorted_list, m=m, args=args, feature_extracor=jrg, rgs = score_rgs, seen_tasks=seen_tasks)
    exemplar_set = save_exemplar(exemplar_set, data2save, task)
    return exemplar_set


def get_m_exemplar_simple(task, packed_sorted_list, m, args, feature_extracor, rgs, seen_tasks):
    data_size = len(packed_sorted_list)   # 数据集的大小
    select_idx = []
    if args.replay_method == 'group_replay':
        if 'debug' in args.exp_name:
            print('group_replay')
        jump = int(data_size / m)              
        idx_list_with_jump = [i for i in range(0,data_size,jump)]
        select_idx = idx_list_with_jump[0:int(m-1)]     
        select_idx.append(idx_list_with_jump[-1])       
    elif args.replay_method == 'random':
        idx_list = np.arange(data_size)
        np.random.shuffle(idx_list)
        select_idx = idx_list[:int(m)].tolist()
    elif args.replay_method == 'herding':
        loader = builder.load_data_from_list(packed_sorted_list, task)
        features2herding, scores2herding = get_features_score(feature_extracor, rgs, loader, seen_tasks, args)
        select_idx = herding(features2herding, scores2herding, m)
    data2save = [packed_sorted_list[j] for j in range(len(packed_sorted_list)) if j in select_idx]

    return data2save

def get_features_score(jrg, rgs, loader, seen_tasks, args):
    combined_feature = torch.Tensor([])
    combined_score = torch.Tensor([])
    for batch_idx, (feat_whole, feat_patch, scores, action_id, data_id) in enumerate(loader):
        # print('step {}'.format(step))
        feat_whole = feat_whole.cuda()
        feat_patch = feat_patch.cuda()
        scores = scores.float()
        action_id = action_id.cuda()
        batch_size = data_id.shape[0]

        feat, _ = run_jrg(jrg, feat_whole, feat_patch, save_graph=args.save_graph, seen_tasks=seen_tasks)
        if args.save_graph:
            feat = feat.transpose(0,1).gather(1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,512)).squeeze(1)     # [B, 512]
        feat = feat.cpu()
        combined_feature = torch.cat((combined_feature, feat), dim=0)
        combined_score = torch.cat((combined_score, scores.reshape(-1)))
    # feature_list = [feature for feature in combined_feature]
    feature_list = combined_feature.cpu()
    score_list = [score for score in combined_score]
    return feature_list, score_list
    

def herding(feature_list, score_list , m, index_list=None, selected_index=None):

    ## print('m =',m)
    assert len(feature_list) >= m
    # 构建一个索引list
    if index_list is None:
        index_list = [i for i in range(len(score_list))]
    if selected_index is None:
        selected_index = []
    ## print('current index list:', index_list)
    ## print('current selected list:', selected_index)
    feature_list = feature_list.cpu()
    # 获取中心feature
    center_feature = torch.mean(feature_list, 0)
    # 计算所有feature 与中心feature之间的距离
    dis_list = [torch.norm((center_feature - feature_list[i])) for i in range(feature_list.shape[0])]
    # 获取离中心feature最近的一个
    min_idx = np.argmin(dis_list)
    selected_index.append(index_list[min_idx])
    new_feature_list = torch.cat([feature_list[:min_idx], feature_list[min_idx+1:]], dim=0)
    new_score_list = score_list[:min_idx] + score_list[min_idx+1:]
    del index_list[min_idx]
    if m > 1:
        herding(new_feature_list, new_score_list, m-1, index_list, selected_index)
    return selected_index

def reduce_exemplar_sets(m, m_p, exemplar_set):
    new_set = [[] for _ in range(6)]
    for a in range(len(exemplar_set)):
        if len(exemplar_set[a]) == 0:
            continue
        jump = int(m_p / m)
        l = [i for i in range(0,m_p,jump)]
        nl = l[0:int(m-1)]
        nl.append(l[-1])

        new_set[a] = [exemplar_set[a][j] for j in range(len(exemplar_set[a])) if j in nl]
        exemplar_set[a] = new_set[a]

# save new data into exemplar set
def save_exemplar(exemplar_set, data2save, save_id):
    exemplar_set[save_id] = data2save
    return exemplar_set

def reconstruct_exemplar_set(exemplar_set):
    # 将exempalr set整合成一个张量
    train_whole_with_memory = torch.Tensor([])
    train_patch_with_memory = torch.Tensor([])
    train_score_with_memory = torch.Tensor([])
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
                train_score_with_memory = torch.cat((train_score_with_memory, torch.tensor(exemplar[2]).float().unsqueeze(0)),dim=0)
                train_action_name_with_memory += [a]
    if len(train_action_name_with_memory) == 0:
        return None
    return train_whole_with_memory, train_patch_with_memory , train_score_with_memory , train_action_name_with_memory

def random_select_exemplar(combined_exemplar, b, count=0):
    train_whole_with_memory, train_patch_with_memory , train_score_with_memory , train_action_name_with_memory = combined_exemplar
    np.random.seed(count)
    select_id = np.random.randint(0, train_score_with_memory.shape[0], size=b)
    select_whole = train_whole_with_memory[select_id]
    select_patch = train_patch_with_memory[select_id]
    select_score = train_score_with_memory[select_id]
    select_action_name = torch.Tensor(train_action_name_with_memory)[select_id]
    return select_whole, select_patch, select_score, select_action_name, select_id


# ========== Funtions for BEST dataset ==========
def get_feature_pseudo_score_(jrg, rgs, loader, seen_tasks, args):
    combined_feature = torch.Tensor([])
    combined_score = torch.Tensor([])
    combined_ids = torch.Tensor([])

    jrg.eval()
    rgs.eval()
    torch.set_grad_enabled(False)
    for batch_idx, (feat_whole, data_id, action_id) in enumerate(loader):
        # print('step {}'.format(step))
        feat_whole = feat_whole.cuda()
        batch_size = data_id.shape[0]
        action_id = action_id.cuda()

        feat, _,  _, _, _= jrg(feat_whole)
        
        if args.save_graph:
            feat = feat.gather(1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,512)).squeeze(1)     # [B, 512]
        pred = rgs(feat)
        feat = feat.cpu()
        pred = pred.cpu()
        combined_feature = torch.cat((combined_feature, feat), dim=0)
        combined_score = torch.cat((combined_score, pred.reshape(-1)))
        combined_ids = torch.cat((combined_ids, data_id.reshape(-1)), dim=0)
    # feature_list = [feature for feature in combined_feature]
    feature_list = combined_feature.cpu()
    score_list = combined_score.cpu()
    id_list = combined_ids.cpu()
    return feature_list, score_list, id_list

def herding_BEST_(feature_list, m, index_list=None, selected_index=None):

    assert len(feature_list) >= m
    # 构建一个索引list
    if index_list is None:
        index_list = [i for i in range(len(feature_list))]
    if selected_index is None:
        selected_index = []

    feature_list = feature_list.cpu()
    center_feature = torch.mean(feature_list, 0)
    dis_list = [torch.norm((center_feature - feature_list[i])) for i in range(feature_list.shape[0])]
    min_idx = np.argmin(dis_list)

    selected_index.append(index_list[min_idx])
    new_feature_list = torch.cat([feature_list[:min_idx], feature_list[min_idx+1:]], dim=0)

    del index_list[min_idx]
    if m > 1:
        herding_BEST_(new_feature_list, m-1, index_list, selected_index)
    return selected_index

def get_data2save_BEST_(task, packed_sorted_list, pure_task_data, m, args, seen_tasks):
    data_size = len(packed_sorted_list)   # 数据集的大小
    select_idx = []
    if args.replay_method == 'group_replay':
        if 'debug' in args.exp_name:
            print('group_replay')
        jump = int(data_size / m)        
        idx_list_with_jump = [i for i in range(0,data_size,jump)]   

        select_idx = idx_list_with_jump[0:int(m-1)]    
        select_idx.append(idx_list_with_jump[-1])      
    elif args.replay_method == 'random':
        idx_list = np.arange(data_size)
        np.random.shuffle(idx_list)
        select_idx = idx_list[:int(m)].tolist()
    elif args.replay_method == 'herding':

        feature2herding = torch.cat([packed_sorted_list[j][0].unsqueeze(0) for j in range(len(packed_sorted_list))])
        print('num video:', data_size)
        print(feature2herding.shape)
        select_idx = herding_BEST_(feature2herding, m)

    print('select indexes: ', select_idx)
    data2save = [[pure_task_data[int(packed_sorted_list[j][2])] , packed_sorted_list[j][1]] for j in range(len(packed_sorted_list)) if j in select_idx]
    return data2save


def after_train_BEST_(t, task, exemplar_set, packed_list, pure_task_data, args, seen_tasks):
    m = args.memory_size / (t+1)
    packed_sorted_list = sorted(packed_list,key=lambda x:x[1])
    if t != 0:
        m_p = int(args.memory_size / t)
        reduce_exemplar_sets(m, m_p, exemplar_set)
    data2save = get_data2save_BEST_(task, packed_sorted_list=packed_sorted_list, pure_task_data=pure_task_data, m=m, args=args, seen_tasks=seen_tasks)
    exemplar_set = save_exemplar(exemplar_set, data2save, task)
    return exemplar_set

def reconstruct_exemplar_set_BEST_(exemplar_set):
    train_whole_with_memory = torch.Tensor([])
    train_score_with_memory = torch.Tensor([])
    train_action_name_with_memory = []
    if exemplar_set is not None:
        for a in range(len(exemplar_set)):
            if len(exemplar_set[a]) == 0:
                continue
            for exemplar in exemplar_set[a]:
                train_whole_with_memory = torch.cat((train_whole_with_memory, torch.tensor(exemplar[0]).float().unsqueeze(0)),dim=0)
                # np.append(train_scores_with_memory, [exemplar[2]],axis=0)
                train_score_with_memory = torch.cat((train_score_with_memory, torch.tensor(exemplar[1]).float().unsqueeze(0)),dim=0)
                train_action_name_with_memory += [a]
    if len(train_action_name_with_memory) == 0:
        return None
    # print(train_whole_with_memory.shape)    # M, 400, 1024
    # print(train_score_with_memory.shape)    # M
    return train_whole_with_memory, train_score_with_memory , train_action_name_with_memory

def random_select_exemplar_BEST_(combined_exemplar, b, count=None):
    train_whole_with_memory, train_score_with_memory , train_action_name_with_memory = combined_exemplar
    if count is not None:
        np.random.seed(count)
    select_id = np.random.randint(0, train_score_with_memory.shape[0], size=b)
    select_whole = train_whole_with_memory[select_id]
    select_score = train_score_with_memory[select_id]
    select_action_name = torch.Tensor(train_action_name_with_memory)[select_id]
    return select_whole, select_score, select_action_name, select_id


def feat_score_aug_BEST_(feature1, feature_list, score1, score_list, aug_scale=0.3, count=None):
    feature_diff = [(feature2 - feature1) for feature2 in feature_list]
    score_diff = [(score2 - score1) for score2 in score_list]

    aug_vector = torch.zeros(feature1.shape).cuda()
    aug_score = torch.zeros(score1.shape).cuda()
    for diff in feature_diff:
        aug_vector += diff
    for diff in score_diff:
        aug_score += diff
    # print('aug_vector.shape',  aug_vector.shape)        # [B, 256]
    # print('aug_score.shape',  aug_score.shape)          # [B]
    if count is not None:
        np.random.seed(count)
    r = np.random.normal(loc=0.0, scale=aug_scale)              
    # print('r:', r)
    aug_feat = feature1 + (r/len(score_diff))* aug_vector
    aug_score = score1 + (r/len(score_diff)) * aug_score
    return aug_feat, aug_score

# -----------------
## AFC
# -----------------
def my_update_importance(jrg, jrg_pre, score_rgs, diff_rgs, loader_train, optimizer, mse, args, seen_tasks):
    jrg.module.reset_importance()
    jrg.module.start_cal_importance()

    jrg.train()
    torch.set_grad_enabled(True)
    for batch_idx, (whole1, whole2, action_id_1, action_id_2) in enumerate(loader_train):
        batch_size = whole1.shape[0]
                
        # data preparing
        whole1 = whole1.cuda()
        whole2 = whole2.cuda()

        feat, featmap, cosine_tensor, l2_tensor, importance = jrg(torch.cat((whole1, whole2), dim=0))
        pred = score_rgs(feat)
        pred1 = pred[0:whole1.shape[0]]
        pred2 = pred[whole1.shape[0]::]
        
        # augmentation of selected previous data
        # if combined_exemplar is not Non
        gama = args.gama
        margin_tensor = (gama - pred1 + pred2)
        margin_tensor[margin_tensor < 0] = 0
        margin_loss = torch.mean(margin_tensor, dim=0)
        loss = margin_loss.cuda()
        loss.backward(retain_graph=True)

    jrg.module.stop_cal_importance()
    jrg.module.normalize_importance()
    return copy.deepcopy(jrg.module.importance.importance)

def update_importance(jrg, jrg_pre, score_rgs, diff_rgs, loader_train, optimizer, mse):
    if len(self._multiple_devices) > 1:
        logger.info("Duplicating model on {} gpus.".format(len(self._multiple_devices)))
        training_network = nn.DataParallel(self._network, self._multiple_devices)
    else:
        training_network = self._network

    training_network.convnet.reset_importance()
    training_network.convnet.start_cal_importance()
    for i, input_dict in enumerate(loader_train):
        inputs, targets = input_dict["inputs"], input_dict["targets"]
        memory_flags = input_dict["memory_flags"]

        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = training_network(inputs)

        logits = outputs["logits"]
        if self._post_processing_type is None:
            scaled_logits = self._network.post_process(logits)
        else:
            scaled_logits = logits * self._post_processing_type
        if self._nca_config:
            nca_config = copy.deepcopy(self._nca_config)
            if self._network.post_processor:
                nca_config["scale"] = self._network.post_processor.factor

            loss = losses.nca(
                logits,
                targets,
                memory_flags=memory_flags,
                **nca_config
            )

        elif self._softmax_ce:
            # Classification loss is cosine + learned factor + softmax:
            loss = F.cross_entropy(scaled_logits, targets)


        loss.backward()

    training_network.convnet.stop_cal_importance()
    training_network.convnet.normalize_importance()
    return 


# ==================== fisher operation =======================
def compute_fisher_matrix_diag(trn_loader, model, score_rgs,  optimizer, mse , args, seen_tasks, sampling_type='max_pred', num_samples=-1):
    torch.set_grad_enabled(True)
    # Store Fisher Information
    fisher = {n: torch.zeros(p.shape).cuda() for n, p in model.named_parameters()
                if p.requires_grad}
    # Compute fisher information for specified number of samples -- rounded to the batch size
    n_samples_batches = (num_samples // trn_loader.batch_size + 1) if num_samples > 0 \
        else (len(trn_loader.dataset) // trn_loader.batch_size)
    # Do forward and backward pass to compute the fisher information
    model.train()
    gama = args.gama
    for batch_idx, (whole1, whole2, action_id_1, action_id_2) in enumerate(trn_loader):
                
        whole1 = whole1.cuda()
        whole2 = whole2.cuda()
        
        feat, _,  _, _, _ = model(torch.cat((whole1, whole2), dim=0))
        pred = score_rgs(feat)
        pred1 = pred[0:whole1.shape[0]]
        pred2 = pred[whole1.shape[0]::]
        mean_pred = torch.mean(pred)
        bias_ = 1e-6

        margin_tensor = (gama - pred1 + pred2)
        margin_tensor[margin_tensor < 0] = 0

        loss = torch.mean(margin_tensor, dim=0)
        optimizer.zero_grad()
        loss.backward() 
        # Accumulate all gradients from loss with regularization
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.pow(2) * len(whole1)
    # Apply mean across all samples
    n_samples = n_samples_batches * trn_loader.batch_size
    fisher = {n: (p / n_samples) for n, p in fisher.items()}
    torch.set_grad_enabled(False)
    return fisher