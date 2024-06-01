import torch
import torch.optim as optim

def get_optim(model_, score_rgs, diff_rgs, args, optim_id=1):
    optimizer3 = optim.Adam([
                {'params': filter(lambda p: p.requires_grad, model_.parameters()), 'lr': args.base_lr * args.lr_factor},
                {'params': score_rgs.parameters()},
                {'params': diff_rgs.parameters()}
            ], lr=args.base_lr, weight_decay=args.weight_decay)
    return optimizer3
