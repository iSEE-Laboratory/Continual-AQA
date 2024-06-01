import torch
import torch.optim as optim

def get_optim(model_, score_rgs, diff_rgs, args, optim_id=1):

    optimizer2 = optim.SGD([
                {'params': filter(lambda p: p.requires_grad, model_.parameters()), 'lr': args.base_lr * args.lr_factor},
                {'params': score_rgs.parameters()},
                {'params': diff_rgs.parameters()}
            ], lr=args.base_lr, weight_decay=args.weight_decay)
    # 基础adam
    optimizer3 = optim.Adam([
                {'params': filter(lambda p: p.requires_grad, model_.parameters()), 'lr': args.base_lr * args.lr_factor},
                {'params': score_rgs.parameters()},
                {'params': diff_rgs.parameters()}
            ], lr=args.base_lr, weight_decay=args.weight_decay)
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
