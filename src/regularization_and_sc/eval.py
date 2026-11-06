import numpy as np
import torch

def accumulate_results(model, data, set_eval=True):
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if set_eval:
        model.eval()
    output_list = []
    label_list = []
    with torch.no_grad():
        for image,label in data:
            image,label = image.to(dev,dtype), label.to(dev)
            label_list.append(label)
            output_list.append(model(image))
    output_list = torch.cat(output_list)
    label_list = torch.cat(label_list)
    return output_list.cpu(),label_list.cpu()   

def test(model, dataloader, metrics_dict:dict, split_name='test', return_logits=False, return_labels=False) -> dict:
    metrics_vals = {}
    logits, labels = accumulate_results(model, dataloader, set_eval=True)
    for name,fn in metrics_dict.items():
        metrics_vals[split_name+' '+name] = torch.as_tensor(fn(logits,labels)).item()
    if return_logits and return_labels: return metrics_vals, logits.cpu(), labels.cpu()
    elif return_logits: return metrics_vals, logits.cpu()
    elif return_labels: return metrics_vals, labels.cpu()
    return metrics_vals

def evaluate_on_dataloader_dict(model, dataloader_dict:dict, metrics_dict:dict, return_logits=False, return_labels=True):
    log_dicts = {}
    logits = {}
    labels = {}
    for split in dataloader_dict.keys():
        if dataloader_dict[split] is None: continue
        log_dict, lgts, lbls = test(model=model, dataloader=dataloader_dict[split], 
                                            metrics_dict=metrics_dict[split], split_name=split, 
                                            return_logits=True, return_labels=True)
        log_dicts.update(log_dict)
        logits.update({split:lgts})
        labels.update({split:lbls})
    
    if return_logits and return_labels: return log_dicts, logits, labels
    if return_logits: return log_dicts, logits
    if return_labels: return log_dicts, labels
    return log_dicts