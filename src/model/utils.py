from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

def get_scheduler(optimizer, max_steps, warmup_steps, min_lr, start_factor=0.01):
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=start_factor, 
        total_iters=warmup_steps
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=(max_steps - warmup_steps), 
        eta_min=min_lr
    )
    
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_steps]
    )
    
    return scheduler