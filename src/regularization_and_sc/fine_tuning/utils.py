import torch

def freeze_modules(modules: list):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

def freeze_model_but_lastlayer(model, freeze_batchnorm=True):
    modules = list(model.children())[:-1]
    freeze_modules(modules)
    if not freeze_batchnorm:
        for m in model.modules(): 
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                for p in m.parameters():
                    p.requires_grad = True

def set_train_mode(model, frozen_modules: list):
    model.train()
    for module in frozen_modules:
        module.eval()

def get_frozen_modules_from_grad(model) -> list:
    """Return the top-most modules whose parameters all have requires_grad=False.

    Modules without parameters (e.g. ReLU, pooling) are only included as part of
    a frozen parent. Once a module is found to be fully frozen, its children are
    not visited, so the returned list has no redundant nested modules.
    """
    frozen_modules = []

    def _visit(module):
        params = list(module.parameters())
        if params and not any(p.requires_grad for p in params):
            frozen_modules.append(module)
            return
        for child in module.children():
            _visit(child)

    _visit(model)
    return frozen_modules

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