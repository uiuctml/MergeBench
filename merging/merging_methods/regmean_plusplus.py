import os
import gc
import shutil
from collections import defaultdict

from tqdm import tqdm

import torch
from torch import nn

from merging_methods.utils import *
from merging_methods.merger import Merger

import sys
sys.path.append('<YOUR PATH HERE>/MergeBench/merging')
from .regmean_utils import *
from taskloader import *


ARCHITECTURE_MODULE_MAP = {
    "LlamaForCausalLM": {
        "transformer_layers_string": "model.layers",
        "embedding_layer_string": "model.embed_tokens",
        "lm_head_string": "lm_head",
    },
    "Gemma2ForCausalLM": {
        "transformer_layers_string": "model.layers",
        "embedding_layer_string": "model.embed_tokens",
        "lm_head_string": "lm_head",
    }
}


class RegMeanPlusPlus(Merger):
    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)
        self.merged_model = self.base_model
        self.merged_model.config.use_cache = False  # Don't need to store past key values as we are not generating text
        self.number_of_layers = self.merged_model.config.num_hidden_layers

        self.num_finetuned_models = len(self.ft_ckpts)

        module_map = ARCHITECTURE_MODULE_MAP[self.base_model.config.architectures[0]]
        self.transformer_layers_string = module_map["transformer_layers_string"]
        self.embedding_layer_string = module_map["embedding_layer_string"]
        self.lm_head_string = module_map["lm_head_string"]

        self.post_init()
        
    def post_init(self):
        # Init the merged model by weight averaging for weights other than the transformer layers, embedding layers, and LM head
        merged_model_param_names = self.merged_model.state_dict().keys()
        merged_model_param_names = [name for name in merged_model_param_names if not (name.startswith(self.transformer_layers_string) or name.startswith(self.embedding_layer_string) or name.startswith(self.lm_head_string))]
        
        for name in tqdm(merged_model_param_names, desc='Init the merged model by weight averaging'):
            merged_param = torch.mean(torch.stack([self.ft_ckpts[i].state_dict()[name] for i in range(self.num_finetuned_models)]), dim=0)
            self.merged_model.state_dict()[name].copy_(merged_param)
        
    def get_first_layer_input(self, model, trainer, dataloader):
        first_layer_input_batch = []

        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
        def hook(module, args, kwargs, output):
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor): kwargs[k] = v.detach().cpu()  # for 'hidden_states' (if it's explicitly passed with keyword arguments), 'attention_mask', 'position_ids', 'cache_position'
                elif isinstance(v, tuple): kwargs[k] = tuple([vv.detach().cpu() for vv in v])  # for 'position_embeddings'
                else: kwargs[k] = v  # for 'past_key_values'
            
            if len(args) > 0:
                first_layer_input_batch.append({"hidden_states": args[0].detach().cpu(), **kwargs})
            else:
                first_layer_input_batch.append(kwargs)

        model.to(trainer.args.device)
        model.config.num_hidden_layers = 1

        handle = eval(f"model.{self.transformer_layers_string}[0]").register_forward_hook(hook, with_kwargs=True)

        total = len(dataloader)
        for inputs in tqdm(dataloader, total=total, desc="Get (merged) model's 1st layer input"):
            inputs = trainer._prepare_inputs(inputs)
            _ = model(**inputs)

        handle.remove()
        
        model.to("cpu")
        model.config.num_hidden_layers = self.number_of_layers

        return first_layer_input_batch

    def forward_layer(self, layer, task_input):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        layer.to(device)

        for batch_idx, inputs in enumerate(task_input):
            inputs = send_inputs_to_device(inputs, device)
            hidden_states = layer(**inputs)
            if type(hidden_states) == tuple:
                hidden_states = hidden_states[0]
            task_input[batch_idx]['hidden_states'] = hidden_states.detach().cpu()

            inputs = send_inputs_to_device(inputs, "cpu")

        layer.to("cpu")

        return task_input
    
    def merge(self, **kwargs):
        reduction = kwargs["reduction"]

        exam_datasets = kwargs["task_names"].split("-")

        save_dir = self.save_path
        gram_dir = os.path.join(save_dir, "regmeanplusplus")
        param_dir = os.path.join(save_dir, "params")

        gram_dirs = [os.path.join(gram_dir, dataset_name) for dataset_name in exam_datasets]
        param_dirs = [os.path.join(param_dir, dataset_name) for dataset_name in exam_datasets]

        # 1. Compute inputs for 1st layer
        task_inputs = defaultdict(list)
        for idx, dataset_name in enumerate(exam_datasets):
            task_loader = TaskLoader(dataset_name, self.merged_model, self.tokenizer)
            trainer = task_loader.trainer
            dataloader = trainer.get_train_dataloader()
            with torch.no_grad():
                task_inputs[dataset_name] = self.get_first_layer_input(
                    self.merged_model, 
                    trainer, 
                    dataloader
                )

            cleanup_task_loader(task_loader)
            torch.cuda.empty_cache()
            gc.collect()
        gc.collect()

        # 2. Merge each layer
        for layer_idx in tqdm(range(self.number_of_layers), desc="Merging layers"):
            # 2.1. Compute grams for each finetuned model
            for idx, dataset_name in enumerate(exam_datasets):
                finetuned_layer = eval(f"self.ft_ckpts[{idx}].{self.transformer_layers_string}")[layer_idx]
                with torch.no_grad():
                    grams = compute_grams(None, finetuned_layer, task_inputs[dataset_name])
                save_tensor_dict(grams, os.path.join(gram_dir, dataset_name)) # contains most (linear) params grams
                save_tensor_dict(finetuned_layer.state_dict(), os.path.join(param_dir, dataset_name)) # contains all params

                finetuned_layer.to("cpu")
                del finetuned_layer, grams
                torch.cuda.empty_cache()
                gc.collect()
            gc.collect()

            layer_param_names = eval(f"self.merged_model.{self.transformer_layers_string}")[layer_idx].state_dict().keys()

            # 2.2. Merge parameters for this layer
            with torch.no_grad():
                gram_module_names = {f[:-3] for f in os.listdir(gram_dirs[0]) if f.endswith(".pt")}
                avg_params = {}
                for name in layer_param_names:
                    h_avged = False
                    if name.endswith('.weight') and not name.startswith('lm_head'):
                        module_name = name[:-len('.weight')]
                        if module_name in gram_module_names:
                            sum_gram, grams = None, None
                            for model_id in range(len(gram_dirs)):
                                param_grams = torch.load(os.path.join(gram_dirs[model_id], module_name + ".pt"), map_location='cpu').detach()
                                param_grams = reduce_non_diag(param_grams, a=reduction) # avoid degeneration
                                param = torch.load(os.path.join(param_dirs[model_id], name + ".pt"), map_location='cpu').detach()
                                gram_m_w = torch.matmul(param_grams, param.transpose(0, 1))
                                if sum_gram is None:
                                    sum_gram = param_grams.clone()
                                    sum_gram_m_ws = gram_m_w.clone()
                                else:
                                    sum_gram.add_(param_grams)
                                    sum_gram_m_ws.add_(gram_m_w)
                                del param_grams, param, gram_m_w
                                gc.collect()
                            sum_gram_f32 = sum_gram.to(dtype=torch.float32)
                            cond_number = torch.linalg.cond(sum_gram_f32)
                            threshold = 1e8 
                            if cond_number > threshold or torch.any(torch.diag(sum_gram_f32) == 0):
                                sum_gram_inv = torch.linalg.pinv(sum_gram_f32).to(dtype=sum_gram_m_ws.dtype)
                            else:
                                sum_gram_inv = torch.inverse(sum_gram_f32).to(dtype=sum_gram_m_ws.dtype)
                            wt = torch.matmul(sum_gram_inv, sum_gram_m_ws)
                            avg_params[name] = wt.transpose(0, 1)
                            h_avged = True
                    
                    if not h_avged: # if not averaged with regmean, then do simple avg
                        filtered_model_params = None
                        for model_id in range(len(gram_dirs)):
                            if not name.startswith('model.embed') and not name.startswith('lm_head'): # embed_tokens.weight have incompatible dimensions due to vocab size difference
                                filtered_model_param = torch.load(os.path.join(param_dirs[model_id], name + ".pt"), map_location='cpu').detach()
                                if filtered_model_params is None:
                                    filtered_model_params = filtered_model_param.clone()
                                else:
                                    filtered_model_params.add_(filtered_model_param)
                                del filtered_model_param
                                gc.collect()
                                avg_params[name] = filtered_model_params.div(len(gram_dirs))

                eval(f"self.merged_model.{self.transformer_layers_string}")[layer_idx].load_state_dict(avg_params, strict=False)
                avg_params = {}
                del avg_params
                
            shutil.rmtree(gram_dir)
            shutil.rmtree(param_dir)

            # 2.3. Compute inputs for next layer
            if layer_idx == self.number_of_layers - 1:
                task_inputs = {}
                del task_inputs
                continue
            
            # # May be just need to update 'hidden_states' for task_inputs[dataset_name]
            # # Check 'past_key_values'
            for idx, dataset_name in enumerate(exam_datasets):
                with torch.no_grad():
                    task_inputs[dataset_name] = self.forward_layer(
                        eval(f"self.merged_model.{self.transformer_layers_string}")[layer_idx], 
                        task_inputs[dataset_name]
                    )
                torch.cuda.empty_cache()
                gc.collect()
            
            gc.collect()

        self.merged_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
