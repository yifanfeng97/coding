import torch

model_dir = 'model_best.pth'
model_new_dir = 'model_best_new.pth'
ckpt = torch.load(model_dir)
ckpt_new = {'model_param_best': ckpt['google_net']}
torch.save(ckpt_new, model_new_dir)