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
    #DataLoaders
    trainloader, valloader, *eval_trainloader = get_train_val_dataloader(
                num_workers=2,
                batch_size=cfg.train.batchsize,
                val_prop=cfg.train.valid_prop,
                random_seed=(cfg.train.fold+1)*cfg.train.seeds.dataset,
                dataloader_generator_seed=(cfg.train.fold+1)*cfg.train.seeds.dataloader,
                val_batchsize=2048,
                eval_trainloader=cfg.train.metrics.get('train') is not None
            )
    if len(eval_trainloader):
        eval_trainloader = eval_trainloader[0]
    else:
        eval_trainloader = None
    
    #Model Initialization
    torch.manual_seed((cfg.train.fold+1)*cfg.train.seeds.weights)
    model = instantiate(cfg.model)
    if cfg.train.get('set_model') is not None: 
        instantiate(cfg.train.set_model, model=model)

    #Optimizer, Scheduler, Loss
    optimizer = instantiate(cfg.train.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.train.scheduler, optimizer=optimizer)
    loss = instantiate(cfg.train.loss)

    #Metrics Dictionary
    metrics_dict = instantiate(cfg.train.metrics)

    #WandB
    os.environ["WANDB_SILENT"] = "true"
    WandB = OmegaConf.to_container(cfg.train.wandb, resolve=True)
    wb = wandb.init(**WandB)

    #Train Loop
    train(model=model,
        trainloader=trainloader, valloader=valloader, 
        criterion=loss, optimizer=optimizer, scheduler=scheduler, 
        warmup_steps=cfg.train.warmup_steps,
        cutmix=cfg.train.cutmix, mixup=cfg.train.mixup, 
        n_steps=cfg.train.n_steps, n_steps_for_eval=cfg.train.n_steps_for_eval, 
        metrics_dict=metrics_dict,
        wb=wb, eval_trainloader=eval_trainloader,
        saving_path=cfg.train.saving_path, file_name=cfg.train.name)

if __name__ == '__main__':
    main()