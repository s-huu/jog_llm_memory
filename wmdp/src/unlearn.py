import torch
from transformers import Trainer
from utils import load_model, TextDatasetGACustom, TextDatasetKLCustom, custom_data_collator, get_batch_loss
import hydra 
import transformers
import os
from peft import LoraConfig, get_peft_model
from pathlib import Path
from omegaconf import OmegaConf
import copy
import torch.nn.functiional as F

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.oracle_model = kwargs.pop('oracle_model').cuda()
        self.loss_type = kwargs.pop('forget_loss')

        super(CustomTrainer, self).__init__(*args, **kwargs)
        # self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        # model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_type == 'grad_ascent':
            input_ids, labels, attention_mask = inputs
            # forward pass
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = -1. * outputs.loss
        elif self.loss_type == 'KL':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            # forward pass
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            # logits = outputs.get("logits")
            forget_loss = -1. * outputs.loss
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)

            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])
            retain_loss = F.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = forget_loss + retain_loss
        elif self.loss_type == 'grad_diff':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            # forward pass
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            # logits = outputs.get("logits")
            forget_loss = -1. * outputs.loss

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss

            loss = forget_loss + retain_loss
        elif self.loss_type == 'npo':
            self.beta = 0.1
            forget_inputs, _ = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)

            forget_loss_current = get_batch_loss(outputs.logits, labels)

            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
            neg_log_ratios = forget_loss_current - forget_loss_oracle

            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 
        else:
            raise Exception("Loss type not supported")

        return (loss, outputs) if return_outputs else loss

    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = -1. * outputs.loss
        return (loss, logits, labels)
    
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="unlearn_custom")
def main(cfg):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    # model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    # model_id = model_cfg["hf_key"]

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    # save the cfg file
    #if master process
    if os.environ.get('LOCAL_RANK') is None or local_rank == 0:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    model_path = cfg.model_path
    model, tokenizer = load_model(model_path)
    oracle_model, _ = load_model(model_path)

    forget_corpora = "bio-forget-corpus,cyber-forget-corpus-safe"
    retain_corpora = "wikitext,wikitext"
    retain_corpora = retain_corpora.split(",")
    forget_corpora = forget_corpora.split(",")

    max_length = 500
    if cfg.forget_loss in ['grad_ascent', 'grad_diff']:
        torch_format_dataset = TextDatasetGACustom(forget_corpora=forget_corpora, retain_corpora=retain_corpora,tokenizer=tokenizer, max_length=max_length)
    else:
        torch_format_dataset = TextDatasetKLCustom(forget_corpora=forget_corpora, retain_corpora=retain_corpora,tokenizer=tokenizer, max_length=max_length)


    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    # --nproc_per_node gives the number of GPUs per = num_devices. take it from torchrun/os.environ
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    print(f"Len dataset: {len(torch_format_dataset)}")
    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")

    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(2, max_steps//10),
            max_steps=max_steps,
            learning_rate=2e-7,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_strategy="steps",
            save_steps=40,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            evaluation_strategy="no",
            deepspeed='config/ds_config.json',
            weight_decay = cfg.weight_decay
        )
    print("Loading from checkpoint")
    model.gradient_checkpointing_enable()
    print(cfg.LoRA.r)
    
    config = LoraConfig(
        r=cfg.LoRA.r, 
        lora_alpha=cfg.LoRA.alpha, 
        target_modules=find_all_linear_names(model), 
        lora_dropout=cfg.LoRA.dropout,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    if cfg.LoRA.r != 0:
        model = get_peft_model(model, config)
    

    trainer = CustomTrainer(
        model=model,
        oracle_model=oracle_model,
        forget_loss=cfg.forget_loss,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        args=training_args,
        data_collator=custom_data_collator,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    oracle_model.config.use_cache = False
    trainer.train()

    #save the model
    if cfg.LoRA.r != 0:
        model = model.merge_and_unload()
        oracle_model = oracle_model.merge_and_unload()


    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

if __name__ == "__main__":
    main()

