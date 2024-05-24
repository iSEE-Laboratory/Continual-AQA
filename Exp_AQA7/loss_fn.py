
import torch
from torch.nn import functional as F

def compute_loss(t, feat, feat_pre, pred_score, label, mse, ce, softmax, args):
    if (t==0) or (args.approach=='finetune'):
        distill_loss = 0
    else:
        soft_feat = softmax(feat)
        soft_feat_pre = softmax(feat_pre)
        distill_loss = ce(soft_feat_pre, soft_feat)

    mse_loss = mse(label, pred_score.reshape(-1, 1))
    return (distill_loss+mse_loss), (mse_loss, distill_loss)

def mse_(pred_score, label, mse):
    mse_loss = mse(label.reshape(-1, 1), pred_score.reshape(-1, 1))
    return mse_loss

def distill_(feat, feat_pre, mse, softmax):
    # soft_feat = feat.softmax(dim=1)
    # soft_feat_pre = feat_pre.softmax(dim=1).detach()
    distill_loss = mse(feat_pre.detach(), feat)
    return distill_loss


def distill_save_graph_(feat_distill, feat_pre_distill, mse, action_id, seen_tasks=[]):
    '''
        feat_distill : 6,B,512
    '''
    batch_size = feat_distill.shape[1]

    distill_save_graph_loss = 0.0
    for task in seen_tasks:
        if task == seen_tasks[-1]:
            break

        distill_save_graph_loss += mse(feat_pre_distill.detach()[task].reshape(-1, 512), feat_distill[task].reshape(-1, 512))
    return distill_save_graph_loss


def st_pod_(featmap_list, featmap_pre_list, temporal_pool=False, norm=True):
    pod_loss = 0.0
    for i, (featmap, featmap_pre) in enumerate(zip(featmap_list, featmap_pre_list)):
        b = featmap.shape[0]
        featmap_w = featmap.sum(5).view(b, -1)
        featmap_pre_w = featmap_pre.sum(5).view(b, -1)
        featmap_h = featmap.sum(4).view(b, -1)
        featmap_pre_h = featmap_pre.sum(4).view(b, -1)
        if temporal_pool:
            featmap_t = featmap.sum(3).view(b, -1)
            featmap_pre_t = featmap_pre.sum(3).view(b, -1)
            a = torch.cat([featmap_w, featmap_h, featmap_t], dim=-1)
            a_pre = torch.cat([featmap_pre_w, featmap_pre_h, featmap_pre_t], dim=-1)
        else:
            a = torch.cat([featmap_w, featmap_h], dim=-1)
            a_pre = torch.cat([featmap_pre_w, featmap_pre_h], dim=-1)
        if norm:
            a = F.normalize(a, dim=1, p=2)
            a_pre = F.normalize(a_pre, dim=1, p=2)
        layer_loss = torch.mean(torch.frobenius_norm(a - a_pre, dim=-1))
        pod_loss += layer_loss
    pod_loss /= len(featmap_list)
    return pod_loss

def graph_distill_(model_, model_pre_, mse=None):
    graph_distill_loss = 0.0
    s1 = model_.module.spatial_mat1.weight
    s2 = model_.module.spatial_mat2.weight
    s3 = model_.module.spatial_mat3.weight
    s4 = model_.module.spatial_mat4.weight
    s1_pre = model_pre_.module.spatial_mat1.weight
    s2_pre = model_pre_.module.spatial_mat2.weight
    s3_pre = model_pre_.module.spatial_mat3.weight
    s4_pre = model_pre_.module.spatial_mat4.weight

    diff1 = torch.add(s1, s1_pre, alpha=-1)
    diff2 = torch.add(s2, s2_pre, alpha=-1)
    diff3 = torch.add(s3, s3_pre, alpha=-1)
    diff4 = torch.add(s4, s4_pre, alpha=-1)
    
    graph_distill_loss += torch.norm(diff1)
    graph_distill_loss += torch.norm(diff2)
    graph_distill_loss += torch.norm(diff3)
    graph_distill_loss += torch.norm(diff4)

    return graph_distill_loss

def ge_graph_distill_(model_, model_pre_, seen_tasks=[],mse=None):
    graph_distill_loss = 0.0
    spatial_graphs = model_.module.spatial_mats * model_.module.joint_graphs
    temporal_graphs = model_.module.temporal_mats * model_.module.joint_graphs
    # print(spatial_graphs.shape) # 6,4,17,17
    general_spatial_graphs = torch.abs(model_.module.general_spatial_mats * model_.module.joint_graphs)
    general_temporal_graphs = torch.abs(model_.module.general_temporal_mats * model_.module.joint_graphs)
    # print(general_temporal_graphs.shape) # 4,17,17
    spatial_graphs = (1-model_.module.alpha) * spatial_graphs + model_.module.alpha * general_spatial_graphs
    temporal_graphs = (1-model_.module.alpha) * temporal_graphs + model_.module.alpha * general_temporal_graphs

    spatial_graphs_pre = model_pre_.module.spatial_mats * model_pre_.module.joint_graphs
    temporal_graphs_pre = model_pre_.module.temporal_mats * model_pre_.module.joint_graphs

    general_spatial_graphs_pre = torch.abs(model_pre_.module.general_spatial_mats * model_pre_.module.joint_graphs)
    general_temporal_graphs_pre = torch.abs(model_pre_.module.general_temporal_mats * model_pre_.module.joint_graphs)

    spatial_graphs_pre = (1-model_pre_.module.alpha) * spatial_graphs + model_pre_.module.alpha * general_spatial_graphs_pre
    temporal_graphs_pre = (1-model_pre_.module.alpha) * temporal_graphs + model_pre_.module.alpha * general_temporal_graphs_pre

    for task in seen_tasks:
        if task == seen_tasks[-1]:
            continue
        graph_distill_loss += mse(spatial_graphs_pre[task].reshape(4, -1), spatial_graphs[task].reshape(4, -1))
        graph_distill_loss += mse(temporal_graphs_pre[task].reshape(4, -1), temporal_graphs[task].reshape(4, -1))
    return graph_distill_loss

if __name__ == '__main__':
    from models.JRG_ASS import ASS_JRG
    model_ = ASS_JRG(whole_size=800, patch_size=800, seg_num=12, joint_num=17, out_dim=1,save_graph=True, mode = '', G_E_graph=True, alpha=0.5)
    
    ge_graph_distill_(model_, None, None, None)