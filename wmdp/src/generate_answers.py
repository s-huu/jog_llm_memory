from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import argparse

def gen_answers(model, raw_question):
    question = tokenizer(raw_question, return_tensors='pt', padding=True).to(0)
    cur_len = len(question.input_ids[0])
    gen_text = model.generate(**question, max_length=cur_len+200, temperature=1, eos_token_id=tokenizer.encode("<｜end▁of▁sentence｜>"))
    gen_text = tokenizer.batch_decode(gen_text[:,cur_len:])

    print(gen_text[0])
    return gen_text[0]

def get_files(q_file):
    f = open(q_file,'r')
    data = f.read()
    all_data = data.split('\n')
    return all_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, default="shengyuanhu/wmdp_relearn_ga_ckpt_10_zephyr")
    parser.add_argument('--save_file', type=str, required=True, default='results/ga_relearn_answers.csv')
    args = parser.parse_args()
    return args

args = parse_args()

model_path_base = "HuggingFaceH4/zephyr-7b-beta"

model_1 = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_flash_attention_2=True
).cuda()


tokenizer = AutoTokenizer.from_pretrained(
    model_path_base, trust_remote_code=True, use_fast=False
)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
tokenizer.mask_token_id = tokenizer.eos_token_id
tokenizer.sep_token_id = tokenizer.eos_token_id
tokenizer.cls_token_id = tokenizer.eos_token_id

all_answers = []

print('Model and tokenizer loaded...')


questions = get_files("data/all_questions.txt")
for i,q in enumerate(questions):
    print(i)
    answer = gen_answers(model_1, [q,""])
    all_answers.append(answer)

df = pd.DataFrame(all_answers)
df.to_csv(args.save_file)