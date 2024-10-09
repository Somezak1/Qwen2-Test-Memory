from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import os
import re
import argparse

device = "cuda" # the device to load the model onto


parser = argparse.ArgumentParser()
parser.add_argument("--max-new-tokens", type=int, default=4)
parser.add_argument('--model', choices=['fp16', 'awq', 'gptq4'], default='fp16')
parser.add_argument("--no-flash-attn", action="store_true")
args = parser.parse_args()


def f(num):
    # for example: 1000000 -> 1,000,000
    if num == 0:
        return "0".rjust(10)
    num = num / 2 ** 20
    return "{:.4f}".format(num).rjust(10)


def get_pr(device):
    info = torch.cuda.list_gpu_processes(device)
    pid = os.getpid()
    pattern = r'process\s+{}\s+uses\s+(\d+\.\d+)\s+MB\s+GPU\s+memory'.format(pid)
    match = re.search(pattern, info, re.IGNORECASE)
    assert match
    pr = float(match.group(1)) * 2 ** 20
    return pr


def record(s):
    ma = torch.cuda.memory_allocated(device)
    mma = torch.cuda.max_memory_allocated(device)
    mr = torch.cuda.memory_reserved(device)
    mmr = torch.cuda.max_memory_reserved(device)

    try:
        pr = get_pr(device)
        print3 = f"[Process]: {f(pr)}"
    except:
        print3 = f""

    print1 = f"[MA]:{f(ma)}                              [MMA]:{f(mma)}                              "
    print2 = f"[MR]:{f(mr)}                              [MMR]:{f(mmr)}                              "

    print("\n\n" + "=" * 90 + s.center(80) + "=" * 90)
    print(print1 + print2 + print3)


if args.model == "fp16":
    model_path = "../models/Qwen2-7B-Instruct"
elif args.model == "awq":
    model_path = "../models/Qwen2-7B-Instruct-AWQ"
elif args.model == "gptq4":
    model_path = "../models/Qwen2-7B-Instruct-GPTQ-Int4"
else:
    raise ValueError

if args.no_flash_attn:
    attn_implementation = 'eager'
else:
    attn_implementation = 'flash_attention_2'


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map=device,
    attn_implementation=attn_implementation
)
config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
config.min_length = args.max_new_tokens + 24
config.max_new_tokens = args.max_new_tokens

record("After Load Model")  # 【1】

tokenizer = AutoTokenizer.from_pretrained(model_path)

record("After Load Tokenizer")  # 【2】

prompt = "北京有哪些好玩的地方？"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
# model_inputs: {
#     'input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
#                           151645,    198, 151644,    872,    198,  68990, 104719, 108257, 103958,
#                           11319, 151645,    198, 151644,  77091,    198]], device='cuda:0'),
#     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')
# }

record("Before model.generate()")  # 【3】
# 【3】 比 【2】 在 memory allocated 部分多了: model_inputs

max_new_tokens = args.max_new_tokens
generated_ids = model.generate(
    model_inputs.input_ids,
    generation_config=config
)

record("After  model.generate()")  # 【26】

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
print("\nInput  tokens: ", model_inputs["input_ids"].size(1))
print("Output tokens: ", len(generated_ids[0]))

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\nInput:")
print(repr(text))
print("\nOutput:")
print(repr(response))
