import torch
import torch.optim as optim

def get_optim(model_, score_rgs, diff_rgs, args, optim_id=1):
    optimizer = optim.Adam(
            [
            {'params':model_.module.general_spatial_mats, 'weight_decay':0, 'lr': 0.01},
            {'params':model_.module.general_temporal_mats, 'weight_decay':0, 'lr': 0.01},
            {'params':model_.module.spatial_mats, 'weight_decay':0, 'lr': 0.01},
            {'params':model_.module.temporal_mats, 'weight_decay':0, 'lr': 0.01},
            {'params':model_.module.spatial_JCWs, 'weight_decay':args.weight_decay},
            {'params':model_.module.temporal_JCWs, 'weight_decay':args.weight_decay},
            {'params':model_.module.encoders_whole.parameters(), 'weight_decay':args.weight_decay},
            {'params':model_.module.encoders_diffwhole.parameters(), 'weight_decay':args.weight_decay},
            {'params':model_.module.encoders_comm0.parameters(), 'weight_decay':args.weight_decay},
            {'params':model_.module.encoders_comm1.parameters(), 'weight_decay':args.weight_decay},
            {'params':model_.module.encoders_diff0.parameters(), 'weight_decay':args.weight_decay},
            {'params':model_.module.encoders_diff1.parameters(), 'weight_decay':args.weight_decay},
            {'params':model_.module.regressor.parameters(), 'weight_decay':args.weight_decay},
            {'params':model_.module.last_fuse.parameters(), 'weight_decay':args.weight_decay},
            {'params': score_rgs.parameters(), 'weight_decay':args.weight_decay},
            {'params': diff_rgs.parameters(), 'weight_decay':args.weight_decay}]
        , lr=args.base_lr)

    optimizer2 = optim.SGD([
                {'params': filter(lambda p: p.requires_grad, model_.parameters()), 'lr': args.base_lr * args.lr_factor},
                {'params': score_rgs.parameters()},
                {'params': diff_rgs.parameters()}
            ], lr=args.base_lr, weight_decay=args.weight_decay)

    optimizer3 = optim.Adam([
                {'params': filter(lambda p: p.requires_grad, model_.parameters()), 'lr': args.base_lr * args.lr_factor},
                {'params': score_rgs.parameters()},
                {'params': diff_rgs.parameters()}
            ], lr=args.base_lr, weight_decay=args.weight_decay)

    optimizer4 = optim.SGD(
        [
            {'params':model_.module.general_spatial_mats, 'weight_decay':0, 'lr': 0.01},
            {'params':model_.module.general_temporal_mats, 'weight_decay':0, 'lr': 0.01},
            {'params':model_.module.spatial_mats, 'weight_decay':0, 'lr': 0.01},
            {'params':model_.module.temporal_mats, 'weight_decay':0, 'lr': 0.01},
            {'params':model_.module.spatial_JCWs, 'weight_decay':args.weight_decay},
            {'params':model_.module.temporal_JCWs, 'weight_decay':args.weight_decay},
            {'params':model_.module.encoders_whole.parameters(), 'weight_decay':args.weight_decay},
            {'params':model_.module.encoders_diffwhole.parameters(), 'weight_decay':args.weight_decay},
            {'params':model_.module.encoders_comm0.parameters(), 'weight_decay':args.weight_decay},
            {'params':model_.module.encoders_comm1.parameters(), 'weight_decay':args.weight_decay},
            {'params':model_.module.encoders_diff0.parameters(), 'weight_decay':args.weight_decay},
            {'params':model_.module.encoders_diff1.parameters(), 'weight_decay':args.weight_decay},
            {'params':model_.module.regressor.parameters(), 'weight_decay':args.weight_decay},
            {'params':model_.module.last_fuse.parameters(), 'weight_decay':args.weight_decay},
            {'params': score_rgs.parameters(), 'weight_decay':args.weight_decay},
            {'params': diff_rgs.parameters(), 'weight_decay':args.weight_decay}]
    , lr=args.base_lr)
    if optim_id == 1:
        print('optim_1')
        return optimizer
    elif optim_id == 2:
        print('optim_2')
        return optimizer2
    elif optim_id == 3:
        print('optim_3')
        return optimizer3
    elif optim_id == 4:
        print('optim_4')
        return optimizer4
