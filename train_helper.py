import torch

def save_ckpt(cfg, model, epoch, best_prec1, optim):
    checkpoint = {'model_param_best': model.state_dict()}
    torch.save(checkpoint, cfg.ckpt_model)

    # save optim state
    optim_state = {
        'epoch': epoch,
        'best_prec1': best_prec1,
        'optim_state_best': optim.state_dict()
    }
    torch.save(optim_state, cfg.ckpt_optim)
    # problem, should we store latest optim state or model, currently, we donot