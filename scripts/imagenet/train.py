import os
import wandb
from regularization_and_sc.train import train
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate

CONFIG_PATH = "/home/pedro-pordeus/regularization_and_sc/conf/imagenet"

@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg:DictConfig) -> None:
    #DataLoaders
    trainloader = instantiate(cfg.train.trainloader)
    valloader = instantiate(cfg.train.valloader)
    eval_trainloader = instantiate(cfg.train.get("eval_trainloader"))
   
    
    #Model Initialization
    model = instantiate(cfg.model)
    instantiate(cfg.train.get('set_model'), model=model)

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
        saving_path=cfg.train.saving_path, file_name=cfg.train.name, num_classes=cfg.train.num_classes)

if __name__ == '__main__':
    main()