import torch

def freeze_modules(modules: list):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

def set_train_mode(model, frozen_modules: list):
    model.train()
    for module in frozen_modules:
        module.eval()

def get_frozen_layers_inception(inception_block, n_unfrozen_layers=1):
    frozen_modules = []
    if n_unfrozen_layers == 0:
        frozen_modules.append(inception_block)
    elif n_unfrozen_layers >= 3:
        frozen_modules = []
    else:
        if 6-3*n_unfrozen_layers > 0: frozen_modules.append(inception_block.b2[:int(6-(3*n_unfrozen_layers))])
        if 9-3*n_unfrozen_layers > 0: frozen_modules.append(inception_block.b3[:int(9-(3*n_unfrozen_layers))])
    return frozen_modules

def get_frozen_children_modules(model, n_unfrozen_layers=1):
    frozen_modules = []
    all_named_children = [(n, m) for n, m in model.named_children()]
    frozen_modules = [m[1] for m in all_named_children[:-n_unfrozen_layers]]
    return frozen_modules

def get_frozen_layers_googlenet(model, n_unfrozen_layers=1):
    frozen_modules = []
    all_named_children = [(n, m) for n, m in model.named_children()]
    if n_unfrozen_layers > 15:
        return print('Too many unfrozen layers')
    if n_unfrozen_layers < 2:
        i = -3 if n_unfrozen_layers == 1 else len(all_named_children)
        frozen_modules = [m[1] for m in all_named_children[:i]]
    else:
        i = 4+((n_unfrozen_layers-2)//3)
        frozen_modules = [m[1] for m in all_named_children[:-i]]
        if len(all_named_children[-i][0]) == 2:
            n = (n_unfrozen_layers-1)%3 if (n_unfrozen_layers-1)%3 != 0 else 3
            frozen_modules.extend(get_frozen_layers_inception(all_named_children[-i][1], n_unfrozen_layers=n))
    return frozen_modules