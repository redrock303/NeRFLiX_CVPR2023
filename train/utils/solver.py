import math

import torch.optim as optim


def make_optimizer(config, model, num_gpu=None):
    if num_gpu is None:
        lr = config.SOLVER.BASE_LR
    else:
        lr = config.SOLVER.BASE_LR * num_gpu

    if config.SOLVER.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,model.parameters()),
                               lr=lr, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=config.SOLVER.WEIGHT_DECAY)
    elif config.SOLVER.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(params=filter(lambda p: p.requires_grad,model.parameters()),
                              lr=lr, momentum=config.SOLVER.MOMENTUM,
                              weight_decay=config.SOLVER.WEIGHT_DECAY)
    elif config.SOLVER.OPTIMIZER == 'Adamax':
        print('adamax used')
        optimizer = optim.Adamax(params=filter(lambda p: p.requires_grad,model.parameters()),
                               lr=lr, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=config.SOLVER.WEIGHT_DECAY)
    else:
        raise ValueError('Illegal optimizer.')

    return optimizer


def make_lr_scheduler(config, optimizer):
    w_iter = config.SOLVER.WARM_UP_ITER
    w_fac = config.SOLVER.WARM_UP_FACTOR
    max_iter = config.SOLVER.MAX_ITER
    lr_lambda = lambda iteration: w_fac + (1 - w_fac) * iteration / w_iter \
        if iteration < w_iter \
        else 1 / 2 * (1 + math.cos((iteration - w_iter) / (max_iter - w_iter) * math.pi))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    return scheduler
