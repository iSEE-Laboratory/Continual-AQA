import torch
from torch.nn import functional as F

def feature_distill_(feat, feat_pre, mse, softmax=None):
    '''
        feat: feature from current model
        feat_pre: feature from previously trained model
    '''

    distill_loss = mse(feat_pre.detach(), feat)
    return distill_loss


def mse_(pred_score, label, mse):
    mse_loss = mse(label.reshape(-1, 1), pred_score.reshape(-1, 1))
    return mse_loss

def pod_(featmap_list, featmap_pre_list, temporal_pool=True, norm=True, args=None, old_importance=None, use_importance=True):
    afc_loss = 0.0
    # print(old_importance[0].shape)
    
    for i, (featmap, featmap_pre, imp) in enumerate(zip(featmap_list, featmap_pre_list, old_importance)):
        bs = featmap.shape[0]

        d = featmap.shape[-1]
        featmap_s = featmap.sum(2).view(bs, d, -1)
        featmap_pre_s = featmap_pre.sum(2).view(bs, d, -1)
        featmap_t = featmap.sum(1).view(bs, d, -1)
        featmap_pre_t = featmap_pre.sum(1).view(bs, d, -1)
        featmap_m = featmap.sum(3).view(bs, d, -1)
        featmap_pre_m = featmap_pre.sum(3).view(bs, d, -1)


        a = torch.cat([featmap_s, featmap_t, featmap_m], dim=-1)
        b = torch.cat([featmap_pre_s, featmap_pre_t, featmap_pre_m], dim=-1)

        if norm:

            a = F.normalize(a, dim=2, p=2)
            b = F.normalize(b, dim=2, p=2)
        if use_importance:
            factor = imp.reshape([1,-1])
        else:
            factor = 1

        layer_loss = torch.mean(factor * torch.frobenius_norm(a - b, dim=-1))
        afc_loss += layer_loss

    return afc_loss / len(featmap_list)

def feature_distill_save_graph_(feat_distill, feat_pre_distill, mse, action_id, seen_tasks=[], args=None):
    '''
        feat_distill : A,B,512
    '''
    batch_size = feat_distill.shape[1]

    distill_save_graph_loss = 0.0
    for task in seen_tasks:
        if task == seen_tasks[-1]:
            break
        # print('compute loss')
        distill_save_graph_loss += mse(feat_pre_distill.detach()[task].reshape(-1, 512), feat_distill[task].reshape(-1, 512))
    return distill_save_graph_loss


def ewc_loss_(fisher, model, older_params, lamb=500):
    loss_reg = 0

    for n, p in model.named_parameters():
        if n in fisher.keys():
            loss_reg += torch.sum(fisher[n] * (p - older_params[n]).pow(2)) / 2
    ewc_loss = lamb * loss_reg

    return ewc_loss