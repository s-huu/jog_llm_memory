This section of the repo contains codes for reproducing the synthetic data experiment in our paper.

## Finetune

We provide the sequences of common names in `common_names_ft.txt`. We constructed twelve distinct sequences of names, with the pair `Anthony Mark` appearing 7 times within the first 4 sequences to enforce a strong correlation. Each sequence is duplicated once in the file.

To finetune a `Llama-2-7b` model that memorizes these sequences of names, run the following commands:
```
master_port=26665
model=llama2-7b
lr=1e-5
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port finetune.py --config-name=finetune.yaml batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}
```

## Unlearn

First replace `model_path` in `config/forget.yaml` to your finetuned model path. We provide an example model that is already finetuned on `common_names_ft.txt`. 

To unlearn sequences from `common_names_ft.txt`, run the following commands:
```
master_port=26665
model=llama2-7b
lr=1e-5
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port forget.py --config-name=forget.yaml batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}
```

## Relearn

First replace `model_path` in `config/relearn.yaml` to your unlearned model path. We provide an example model that is already unlearned using grad ascent. 

To perform relearning, we provide the sequences of common names up to the name `Anthony` in `common_names_relearn.txt`. These are all subsequences from the unlearn set. Run the following command:
```
master_port=26665
model=llama2-7b
lr=1e-5
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port relearn.py --config-name=relearn.yaml batch_size=2 gradient_accumulation_steps=8 model_family=${model} lr=${lr}
```

## Evaluate

In `eval.py`, we generate 100 random queries ending with token `Anthony` and evaluate whether `Anthony Mark` will appear in the model output. We provide an example checkpoint where relearning is successful.
```
python eval.py
```
