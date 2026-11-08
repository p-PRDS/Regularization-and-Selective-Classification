import pandas as pd 
import torch 
import os
from regularization_and_sc.imagenet.data import get_dataloader
from regularization_and_sc.eval import evaluate_on_dataloader_dict
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

CONFIG_PATH = "../../conf/imagenet"
@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg:DictConfig) -> None:
    model = instantiate(cfg.model)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    eval_loaders = instantiate(cfg.eval.loaders, _convert_="partial")
    metrics_dict = instantiate(cfg.eval.metrics)
    metrics_df = pd.DataFrame()
    log_dicts, logits, labels = evaluate_on_dataloader_dict(model, eval_loaders, metrics_dict, return_logits=True, return_labels=True)

    os.makedirs(f"{cfg.eval.model.path}/labels", exist_ok=True)
    for split in eval_loaders.keys():
        os.makedirs(f"{cfg.eval.model.path}/logits/{split}", exist_ok=True)
        torch.save(logits[split], f"{cfg.eval.model.path}/logits/{split}/{cfg.eval.model.weights_name}_fold_{cfg.fold}.pt")
        torch.save(labels[split], f"{cfg.eval.model.path}/labels/{split}_labels_fold_{cfg.fold}.pt")

    if os.path.exists(f"{cfg.eval.model.path}/eval_dfs/metrics_df.pkl"):
        metrics_df = pd.read_pickle(f"{cfg.eval.model.path}/eval_dfs/metrics_df.pkl")
    metrics_df.loc[cfg.fold] = log_dicts
    print(metrics_df)
    os.makedirs(f"{cfg.eval.model.path}/eval_dfs", exist_ok=True)    
    metrics_df.to_pickle(f"{cfg.eval.model.path}/eval_dfs/metrics_df.pkl")

if __name__ == "__main__":
    main()