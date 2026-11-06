import os
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from regularization_and_sc.cifar.utils import get_network, get_train_val_dataloader
from regularization_and_sc.train import train
from regularization_and_sc.utils import metrics, measures
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate, get_method

CONFIG_PATH = "/home/pedro-pordeus/regularization_and_sc/conf/cifar"

@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg:DictConfig) -> None:
    for fold in range(cfg.train.folds):
        #DataLoaders
        trainloader, valloader, *eval_trainloader = get_train_val_dataloader(
                    num_workers=2,
                    batch_size=cfg.train.batchsize,
                    val_prop=cfg.train.valid_prop,
                    random_seed=(fold+1)*cfg.train.data_seed,
                    dataloader_generator_seed=(fold+1)*cfg.train.dataloader_seed,
                    val_batchsize=2048,
                    eval_trainloader=cfg.train.metrics_dict.get('train') is not None
                )
        if len(eval_trainloader):
            eval_trainloader = eval_trainloader[0]
        else:
            eval_trainloader = None
        
        #Model Initialization
        torch.manual_seed((fold+1)*cfg.train.model_seed)
        model = get_network(cfg.model.name)
        if cfg.model.get('frozen_modules') is not None:
            frozen_modules = instantiate(cfg.model.frozen_modules)(model)
        else:
            frozen_modules = []
    
        if cfg.train.get('pretrained_weights'):
            model.load_state_dict(torch.load(f"{cfg.train.pretrained_weights.path}/{cfg.train.pretrained_weights.name}_fold_{fold}_{cfg.train.pretrained_weights.checkpoint}.pt"))

        #Optimizer, Scheduler, Criterion
        optimizer = instantiate(cfg.train.optimizer, params=model.parameters())
        if cfg.train.get('scheduler'):
            scheduler = instantiate(cfg.train.scheduler, optimizer=optimizer)
        else:
            scheduler = None
        criterion = instantiate(cfg.train.criterion)

        #Metrics Dictionary
        metrics_dict = {}
        for split in cfg.train.metrics_dict.keys():
            metrics_dict[split] = {}
            for metric in cfg.train.metrics_dict[split]:
                if "FC-AURC" in metric:
                    metrics_dict[split][metric] = instantiate(cfg.train.metrics_dict[split][metric], 
                                                             classifier_logits_path=f"{cfg.train.classifier.path}/logits/{split}/{cfg.train.classifier.name}_fold_{fold}_{cfg.train.classifier.checkpoint}.pt")
                else:
                    metrics_dict[split][metric] = instantiate(cfg.train.metrics_dict[split][metric])
            metrics_dict[split]['loss'] = criterion

        #WandB
        os.environ["WANDB_SILENT"] = "true"
        name = f"{cfg.train.wandb.group}_fold_{fold}"
        WandB = OmegaConf.to_container(cfg.train.wandb, resolve=True)
        WandB['name'] = name
        wb = wandb.init(**WandB)

        #Train Loop
        train(model=model, frozen_modules=frozen_modules,
            trainloader=trainloader, valloader=valloader, 
            criterion=criterion, optimizer=optimizer, scheduler=scheduler, 
            warmup_steps=cfg.train.warmup_steps,
            cutmix=cfg.train.cutmix, mixup=cfg.train.mixup, 
            n_steps=cfg.train.n_steps, n_steps_for_eval=cfg.train.n_steps_for_eval, 
            metrics_dict=metrics_dict,
            wb=wb, eval_trainloader=eval_trainloader,
            saving_path=cfg.train.saving_path, file_name=name)

if __name__ == '__main__':
    main()