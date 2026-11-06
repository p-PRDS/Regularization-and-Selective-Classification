import os
import wandb
import numpy as np
import torch
from torchvision.transforms import v2
from regularization_and_sc.eval import test, evaluate_on_dataloader_dict
from .cifar.utils import WarmUpLR
from .fine_tuning.utils import set_train_mode, freeze_modules

def apply_mixup_cutmix(images, labels, mixup=0, cutmix=0):
    if mixup: 
        if cutmix: 
            preproc = v2.RandomChoice([v2.CutMix(alpha=cutmix,num_classes=100), v2.MixUp(alpha=mixup,num_classes=100)])
        else:
            preproc = v2.MixUp(alpha=mixup,num_classes=100)
    elif cutmix:
        preproc = v2.CutMix(alpha=cutmix,num_classes=100)
    else:
        return images, labels
    return preproc(images, labels)

def train(model, trainloader, valloader, criterion, optimizer, scheduler=None, warmup_steps=0,
        cutmix=0, mixup=0, n_steps=100, n_steps_for_eval=100, metrics_dict:dict=None, wb=None, eval_trainloader=None,
        saving_path='.', file_name='model', frozen_modules=[]):
    if not os.path.exists(f'{saving_path}/weights'):
        os.makedirs(f'{saving_path}/weights')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev)
    freeze_modules(frozen_modules)
    eval_loaders = {'val':valloader, 'train':eval_trainloader} if eval_trainloader else {'val':valloader}
    best = {'Accuracy':0, 'NLL':9999, 'FC-AURC':1, 'AURC':1}

    warmup_scheduler = WarmUpLR(optimizer, warmup_steps)

    #First evaluation on step 0
    log_dicts = evaluate_on_dataloader_dict(model, eval_loaders, metrics_dict, return_logits=False, return_labels=False)
    if wb: wb.log(log_dicts)

    #Training
    step = 0
    while step < n_steps:
        set_train_mode(model, frozen_modules)
        for images, labels in trainloader:
            images, labels = apply_mixup_cutmix(images, labels, mixup, cutmix)
            images = images.to(dev)
            labels = labels.to(dev)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            if step < warmup_steps:
                warmup_scheduler.step()
            step+=1
        
            #Evaluation
            if step % n_steps_for_eval == 0:
                with torch.no_grad():
                    log_dicts = evaluate_on_dataloader_dict(model, eval_loaders, metrics_dict, return_logits=False, return_labels=False)
                    if wb: 
                        wb.log(log_dicts)
                    
                    #Checkpoints Saving
                    if log_dicts.get(f'val Accuracy') and log_dicts['val Accuracy'] > best['Accuracy']:
                        torch.save(model.state_dict(), f'{saving_path}/weights/{file_name}_bestAccuracy.pt')
                        best['Accuracy'] = log_dicts['val Accuracy']

                    if log_dicts.get(f'val NLL') and log_dicts['val NLL'] < best['NLL']:
                        torch.save(model.state_dict(), f'{saving_path}/weights/{file_name}_bestNLL.pt')
                        best['NLL'] = log_dicts['val NLL']

                    if log_dicts.get(f'val FC-AURC') and log_dicts['val FC-AURC'] < best['FC-AURC']:
                        torch.save(model.state_dict(), f'{saving_path}/weights/{file_name}_bestFCAURC.pt')
                        best['FC-AURC'] = log_dicts['val FC-AURC']
                    
                    if log_dicts.get(f'val AURC') and log_dicts['val AURC'] < best['AURC']:
                        torch.save(model.state_dict(), f'{saving_path}/weights/{file_name}_bestAURC.pt')
                        best['AURC'] = log_dicts['val AURC']
                set_train_mode(model, frozen_modules)
            if scheduler and step >= warmup_steps: scheduler.step()
    torch.save(model.state_dict(), f'{saving_path}/weights/{file_name}_LastStep.pt')
    if wb: wb.finish()
