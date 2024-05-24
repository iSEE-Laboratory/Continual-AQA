import os,sys
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm
import time
import scipy
import builder
from models.JRG_ASS import run_jrg
# from models.JRG_ASS import run_jrg

# ======================================================== 
def get_model(model):
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


# ========================================
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

# ================= FS-Aug =======================
def feat_score_aug(feature1, feature_list, score1, score_list, aug_scale=0.3, count=None, with_weight=False):
    """
        b: batch size
        d: feature dimension
        n: number of helpers
        feature1: [b, d]
        feature_list: [n, d]
        score1: [b]
        score_list: [n]
    """
    feature_diff = [(feature2 - feature1) for feature2 in feature_list]
    score_diff = [(score2 - score1) for score2 in score_list]


    if not with_weight:
        aug_vector = torch.zeros(feature1.shape)
        aug_score = torch.zeros(score1.shape)
        for diff in feature_diff:
            aug_vector += diff
        for diff in score_diff:
            aug_score += diff

        if count is not None:
            np.random.seed(count)
        r = np.random.normal(loc=0.0, scale=aug_scale)

        aug_feat = feature1 + (r/len(score_diff))* aug_vector
        aug_score = score1 + (r/len(score_diff)) * aug_score
    else:
        sorted_indices = torch.argsort(torch.cat([d.unsqueeze(1) for d in score_diff], dim=1))
        ranking = torch.argsort(sorted_indices)        
        weights = np.linspace(0.1, 1, num=len(feature_list))
        weight_matrix = torch.zeros(ranking.shape)
        for i in range(ranking.shape[0]):
            for j in range(ranking.shape[1]):
                weight_matrix[i][j] = weights[ranking[i][j]]
        aug_vector = torch.zeros(feature1.shape)
        aug_score = torch.zeros(score1.shape)
        
        for i, diff in enumerate(feature_diff):
            weight_i = weight_matrix[:, i]
            aug_vector += (weight_i.unsqueeze(1).expand(-1, diff.shape[1]) * diff)
        for i, diff in enumerate(score_diff):
            weight_i = weight_matrix[:, i]
            aug_score += (weight_i) * diff
        if count is not None:
            np.random.seed(count)
        r = np.random.normal(loc=0.0, scale=aug_scale)

        aug_feat = feature1 + (r/torch.sum(weight_matrix, dim=1).unsqueeze(1).expand(-1, aug_vector.shape[1]))* aug_vector
        aug_score = score1 + (r/torch.sum(weight_matrix, dim=1)) * aug_score
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
    data_size = len(packed_sorted_list) 
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
    if index_list is None:
        index_list = [i for i in range(len(score_list))]
    if selected_index is None:
        selected_index = []
    ## print('current index list:', index_list)
    ## print('current selected list:', selected_index)
    feature_list = feature_list.cpu()
    center_feature = torch.mean(feature_list, 0)
    dis_list = [torch.norm((center_feature - feature_list[i])) for i in range(feature_list.shape[0])]
    min_idx = np.argmin(dis_list)
    
    ## print('get min index:', index_list[min_idx])
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