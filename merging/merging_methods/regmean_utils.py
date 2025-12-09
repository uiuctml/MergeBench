
import re
import os
from tqdm import tqdm
import torch
from torch import nn
import gc

# https://github.com/bloomberg/dataless-model-merging/blob/main/regmean_demo.ipynb

def filter_modules_by_regex(base_module, include_patterns, include_type):
    modules = {}
    for name, module in base_module.named_modules():
        valid_name = not include_patterns or any(
            [re.match(patt, name) for patt in include_patterns]
        )
        valid_type = not include_type or any(
            [isinstance(module, md_cls) for md_cls in include_type]
        )
        if valid_type and valid_name:
            modules[name] = module
    return modules


def send_inputs_to_device(inputs, device):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor): inputs[k] = v.to(device)  # for 'hidden_states', 'attention_mask', 'position_ids', 'cache_position', 'past_key_values'
        elif isinstance(v, tuple): inputs[k] = tuple([vv.to(device) for vv in v])  # for 'position_embeddings'

    return inputs


def compute_grams(trainer, finetuned_model, train_dataloader):
    covs = {}
    xn = {}

    def get_grams(name):
        def hook(module, input, output):
            """
            Note: adhere to signature of hook functions
            """
            x = input[0].detach()  # $[b,t,h]
            x = x.view(-1, x.size(-1))
            xtx = torch.matmul(x.transpose(0, 1), x)  # [h,h]
            if name not in covs:
                covs[name] = xtx / x.size(0)
                xn[name] = x.size(0)
            else:
                covs[name] = (covs[name] * xn[name] + xtx) / (x.size(0) + xn[name])
                xn[name] += x.size(0)

        return hook

    device = "cpu"
    if trainer is not None:
        device = trainer.args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    model = finetuned_model.to(device)
    linear_modules = filter_modules_by_regex(
        model, None, [nn.Linear]
    )
    handles = []
    for name, module in linear_modules.items():
        handle = module.register_forward_hook(get_grams(name))
        handles.append(handle)

    total = len(train_dataloader)
    for inputs in tqdm(
        train_dataloader, total=total, desc="Computing gram matrix",
        disable = type(train_dataloader) == list
    ):
        if type(train_dataloader) == list:
            # For RegMeanPlusPlus
            inputs = send_inputs_to_device(inputs, device)
            _ = model(**inputs)
            inputs = send_inputs_to_device(inputs, "cpu")
        else:
            # For RegMean
            inputs = trainer._prepare_inputs(inputs)
            _ = model(**inputs)

    for handle in handles:
        handle.remove()

    return covs

def reduce_non_diag(cov_mat, a):
    diag_weight = torch.diag(torch.ones(cov_mat.size(0), dtype=cov_mat.dtype) - a).to(cov_mat.device)
    non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
    weight = diag_weight + non_diag_weight
    return cov_mat * weight

def save_tensor_dict(tensor_dict, path):
    os.makedirs(path, exist_ok=True)
    for key, tensor in tensor_dict.items():
        torch.save(tensor, os.path.join(path, key + ".pt"))

def cleanup_task_loader(task_loader):
    """
    Safely clean up task_loader and its nested trainer to reduce CPU memory usage.
    """
    trainer = getattr(task_loader, 'trainer', None)

    if trainer is not None:
        for attr in [
            'model', 'processing_class', 'train_dataset', 'eval_dataset',
            'callback_handler', 'args', 'data_collator',
            'train_dataloader', 'eval_dataloader',
            'optimizer', 'lr_scheduler',
        ]:
            if hasattr(trainer, attr):
                try:
                    setattr(trainer, attr, None)
                except Exception as e:
                    print(f"Warning: couldn't clear trainer.{attr}: {e}")
        del trainer

    for attr in ['training_dataset', 'training_args']:
        if hasattr(task_loader, attr):
            try:
                setattr(task_loader, attr, None)
            except Exception as e:
                print(f"Warning: couldn't clear task_loader.{attr}: {e}")
    del task_loader

    gc.collect()