import numpy as np
import pandas as pd 
import torch 
import os

from regularization_and_sc.cifar.utils import get_network, get_train_val_dataloader, get_test_dataloader
from regularization_and_sc.utils import metrics, measures
from regularization_and_sc.eval import evaluate_on_dataloader_dict, test

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate

CONFIG_PATH = "/home/pedro-pordeus/regularization_and_sc/conf/cifar"
@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg:DictConfig) -> None:
    model = get_network(cfg.model.name)
    metrics_df = pd.DataFrame()
    for fold in cfg.eval.folds:
        _, valloader, *eval_trainloader = get_train_val_dataloader(
                            num_workers=2,
                            val_prop=cfg.eval.valid_prop,
                            random_seed=(fold+1)*cfg.eval.data_seed,
                            val_batchsize=2048,
                            eval_trainloader=cfg.eval.metrics_dict.get('train') is not None
                        )

        testloader = get_test_dataloader(batch_size=2048, num_workers=2)

        model.load_state_dict(torch.load(f"{cfg.eval.model.path}/weights/{cfg.eval.model.weights_name}_fold_{fold}_{cfg.eval.model.checkpoint}.pt"))
        model.to('cuda' if torch.cuda.is_available() else 'cpu')

        eval_loaders = {'val':valloader, 'test':testloader, 'train':eval_trainloader[0]} if len(eval_trainloader) else {'val':valloader, 'test':testloader}

        metrics_dict = {}
        for split in cfg.eval.metrics_dict.keys():
            metrics_dict[split] = {}
            for metric in cfg.eval.metrics_dict[split]:
                if "FC-AURC" in metric:
                    metrics_dict[split][metric] = instantiate(cfg.eval.metrics_dict[split][metric], 
                                                                classifier_logits_path=f"{cfg.eval.classifier_path}/logits/{split}/{cfg.eval.classifier_name}_fold_{fold}.pt")
                else:
                    metrics_dict[split][metric] = instantiate(cfg.eval.metrics_dict[split][metric])

        log_dicts, logits, labels = evaluate_on_dataloader_dict(model, eval_loaders, metrics_dict, return_logits=True, return_labels=True)
        if not os.path.exists(f"{cfg.eval.model.path}/labels"):
            os.makedirs(f"{cfg.eval.model.path}/labels")
        for split in eval_loaders.keys():
            if not os.path.exists(f"{cfg.eval.model.path}/logits/{split}"):
                os.makedirs(f"{cfg.eval.model.path}/logits/{split}")
            torch.save(logits[split], f"{cfg.eval.model.path}/logits/{split}/{cfg.eval.model.weights_name}_fold_{fold}_{cfg.eval.model.checkpoint}.pt")
            torch.save(labels[split], f"{cfg.eval.model.path}/labels/{split}_labels_fold_{fold}.pt")
        for metric in log_dicts.keys():
            metrics_df.loc[fold, f'{metric}'] = log_dicts[metric]

    if not os.path.exists(f"{cfg.eval.model.path}/eval_dfs"):
        os.makedirs(f"{cfg.eval.model.path}/eval_dfs")
    metrics_df.to_pickle(f"{cfg.eval.model.path}/eval_dfs/metrics_df.pkl")

if __name__ == "__main__":
    main()