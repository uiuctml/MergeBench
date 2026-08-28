import os

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

cache_dir = os.getenv('HF_HOME', '/data/huggingface')

def formatting_prompts_func(examples, instruction_key='instruction', input_key='input', output_key='output'):
    # alpaca style prompts
    # also works for gpteacher because gpteacher inherits alpaca prompt
    # https://github.com/huggingface/trl/pull/444#issue-1760952763
    instruction = examples[instruction_key]
    if 'input' in examples:
        input_text = examples[input_key]
    else:
        input_text = ''
    response = examples[output_key]

    if len(input_text) > 0:
        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        
        ### Instruction:
        {instruction}
        
        ### Input:
        {input_text}
        
        ### Response:
        {response}
        '''
    else:
        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        
        ### Instruction:
        {instruction}
        
        ### Response:
        {response}
        '''

    return text


class TaskLoader:
    # Category name shared by a task's fine-tuned checkpoint suffix and its
    # validation set (e.g. 'math' -> checkpoint '<model>_math', data 'MergeBench/math_val').
    # Overridden by each concrete task; this is the single source of truth that keeps
    # the merged checkpoint, its Gram statistics, and the task ordering aligned.
    category = None

    def __new__(cls, task_name, *args, **kwargs):
        if task_name in globals() and issubclass(globals()[task_name], cls):
            subclass = globals()[task_name]
            return super().__new__(subclass)
        else:
            raise ValueError(f"Invalid task name: {task_name}")

    def __init__(self, task_name, *args, **kwargs):
        self.task_name = task_name

    @classmethod
    def checkpoint_for(cls, model_name):
        return f'MergeBench/{model_name}_{cls.category}'


# Canonical task ordering, used as a fallback for merging algorithms that do not
# take an explicit --task_names argument (their operation is symmetric across models,
# so the order does not affect correctness).
DEFAULT_TASK_ORDER = ['Tulu3IF', 'DartMath', 'MagiCoder', 'WildguardMix', 'Aya']


def get_ft_ckpts(base_model, task_names=None):
    """Derive the fine-tuned checkpoint paths from the task ordering.

    task_names is the ordered list of TaskLoader class names (e.g. from
    --task_names.split('-')). Deriving the checkpoints from the same list the
    merge loop iterates guarantees each checkpoint lines up with the dataset used
    to compute its Gram statistics. Falls back to DEFAULT_TASK_ORDER when no
    task_names are given.
    """
    model_name = base_model.split('/')[-1]
    task_names = task_names or DEFAULT_TASK_ORDER
    return [globals()[task_name].checkpoint_for(model_name) for task_name in task_names]


class WildguardMix(TaskLoader):
    category = 'safety'

    def __init__(self, task_name, model, tokenizer, sample_size=None):
        super().__init__(task_name, model, tokenizer, sample_size=sample_size)

        self.training_args = SFTConfig(
                            learning_rate=1e-5,
                            num_train_epochs=1,
                            lr_scheduler_type='cosine',
                            optim="adamw_torch",
                            bf16=True,
                            dataset_num_proc=48,
                            packing=False,
                            max_length=2048, # 4096
                            gradient_checkpointing=True,
                            per_device_train_batch_size=1,
                            # deepspeed='/home/cindy2000_sh/MergeBench/deepspeed_configs/zero3.json',
                            output_dir="./tmp",
                            save_strategy='no',
                        ) 

        self.training_dataset = load_dataset(f'MergeBench/{self.category}_val',cache_dir=cache_dir)
        self.training_dataset = self.training_dataset.rename_column("prompt", "query")
        
        if sample_size is None:
            self.training_dataset = self.training_dataset["train"]
        else:
            self.training_dataset = self.training_dataset["train"].shuffle(seed=42).select(range(sample_size))                     
        self.trainer = SFTTrainer(model=model,
                                  args=self.training_args,
                                  train_dataset=self.training_dataset,  
                                  formatting_func=lambda examples: formatting_prompts_func(
                                    examples, instruction_key="query", output_key="response"
                                ),
                            )


class MagiCoder(TaskLoader):
    category = 'coding'

    def __init__(self, task_name, model, tokenizer, sample_size=None):
        super().__init__(task_name, model, tokenizer, sample_size=sample_size)

        self.training_args = SFTConfig(
                            learning_rate=1e-5,
                            num_train_epochs=1,
                            lr_scheduler_type='cosine',
                            optim="adamw_torch",
                            bf16=True,
                            dataset_num_proc=48,
                            packing=False,
                            max_length=2048, # 4096
                            gradient_checkpointing=True,
                            per_device_train_batch_size=1,
                            # deepspeed='/home/cindy2000_sh/MergeBench/deepspeed_configs/zero3.json',
                            output_dir="./tmp",
                            save_strategy='no',
                        ) 

        self.training_dataset = load_dataset(f'MergeBench/{self.category}_val',cache_dir=cache_dir)

        if sample_size is None:
            self.training_dataset = self.training_dataset["train"]
        else:
            self.training_dataset = self.training_dataset["train"].shuffle(seed=42).select(range(sample_size))                
        self.trainer = SFTTrainer(model=model,
                                  args=self.training_args,
                                  train_dataset=self.training_dataset,  
                                  formatting_func=lambda examples: formatting_prompts_func(
                                    examples, output_key="response"
                                ),
                            )


class Aya(TaskLoader):
    category = 'multilingual'

    # TODO: match with Yuzheng's config
    def __init__(self, task_name, model, tokenizer, sample_size=None):
        super().__init__(task_name, model, tokenizer, sample_size=sample_size)

        self.training_args = SFTConfig(
                            learning_rate=2e-5,
                            num_train_epochs=1,
                            lr_scheduler_type='cosine',
                            optim="adamw_torch",
                            bf16=True,
                            dataset_num_proc=48,
                            packing=False,
                            max_length=2048, 
                            gradient_checkpointing=True,
                            per_device_train_batch_size=1,
                            # deepspeed='/home/cindy2000_sh/MergeBench/deepspeed_configs/zero3.json',
                            output_dir="./tmp",
                            save_strategy='no',
                        ) 

        self.training_dataset = load_dataset(f'MergeBench/{self.category}_val',cache_dir=cache_dir)
        if sample_size is None:
            self.training_dataset = self.training_dataset["train"]
        else:
            self.training_dataset = self.training_dataset["train"].shuffle(seed=42).select(range(sample_size))                
        self.trainer = SFTTrainer(model=model,
                                  args=self.training_args,
                                  train_dataset=self.training_dataset,  
                                  formatting_func=lambda examples: formatting_prompts_func(
                                    examples, instruction_key="inputs", output_key="targets"
                                ),
                            )
        

class DartMath(TaskLoader):
    category = 'math'

    def __init__(self, task_name, model, tokenizer, sample_size=None):
        super().__init__(task_name, model, tokenizer, sample_size=sample_size)

        self.training_args = SFTConfig(
                            learning_rate=1e-5,
                            num_train_epochs=1,
                            lr_scheduler_type='cosine',
                            optim="adamw_torch",
                            bf16=True,
                            dataset_num_proc=48,
                            packing=False,
                            max_length=2048, 
                            gradient_checkpointing=True,
                            per_device_train_batch_size=1,
                            # deepspeed='/home/cindy2000_sh/MergeBench/deepspeed_configs/zero3.json',
                            output_dir="./tmp",
                            save_strategy='no',
                        ) 

        self.training_dataset = load_dataset(f'MergeBench/{self.category}_val',cache_dir=cache_dir)

        if sample_size is None:
            self.training_dataset = self.training_dataset["train"]
        else:
            self.training_dataset = self.training_dataset["train"].shuffle(seed=42).select(range(sample_size))                
        self.trainer = SFTTrainer(model=model,
                                  args=self.training_args,
                                  train_dataset=self.training_dataset,  
                                  formatting_func=lambda examples: formatting_prompts_func(
                                    examples, instruction_key="query", output_key="response"
                                ),
                            )
                            
class Tulu3IF(TaskLoader):
    category = 'instruction'

    def __init__(self, task_name, model, tokenizer, sample_size=None):
        super().__init__(task_name, model, tokenizer, sample_size=sample_size)

        self.training_args = SFTConfig(
                            learning_rate=1e-5,
                            num_train_epochs=1,
                            lr_scheduler_type='cosine',
                            optim="adamw_torch",
                            bf16=True,
                            dataset_num_proc=48,
                            packing=False,
                            max_length=2048, 
                            gradient_checkpointing=True,
                            per_device_train_batch_size=1,
                            # deepspeed='/home/cindy2000_sh/MergeBench/deepspeed_configs/zero3.json',
                            output_dir="./tmp",
                            save_strategy='no',
                        ) 

        self.training_dataset = load_dataset(f'MergeBench/{self.category}_val',cache_dir=cache_dir)

        if sample_size is None:
            self.training_dataset = self.training_dataset['train']
        else:
            self.training_dataset = self.training_dataset['train'].shuffle(seed=42).select(range(sample_size))    

                
        self.trainer = SFTTrainer(model=model,
                                  args=self.training_args,
                                  train_dataset=self.training_dataset,  
                                  formatting_func=lambda examples: formatting_prompts_func(
                                    examples
                                ),
                            )

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", cache_dir=cache_dir)

    task_preprocessor = TaskLoader('WildguardMix', model, tokenizer, sample_size=None)
    # task_preprocessor = TaskLoader('MagiCoder', model, tokenizer, sample_size=None)
    # task_preprocessor = TaskLoader('Aya', model, tokenizer, sample_size=None)
    # task_preprocessor = TaskLoader('DartMath', model, tokenizer, sample_size=None)
    # task_preprocessor = TaskLoader('Tulu3IF', model, tokenizer, sample_size=None)
