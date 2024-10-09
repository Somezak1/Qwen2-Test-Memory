import re
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--max-new-tokens", type=int, default=4)
args = parser.parse_args()


def extract(file):
    with open(file, "r") as f:
        lines = f.readlines()

    endpoints = []
    ma = []
    mma = []
    mr = []
    t = []

    for line in lines:
        if line.startswith("======"):
            title = line.replace("=", "").rstrip("\n").strip(" ")
            endpoints.append(title)
        elif line.startswith("[MA]:"):
            match = re.match(r'\[MA\]:\s?(\d+\.\d+)', line)
            assert match, print(line)
            ma.append(float(match.groups()[0]))

            match = re.search(r'\[MMA\]:\s?(\d+\.\d+)', line)
            assert match, print(line)
            mma.append(float(match.groups()[0]))

            match = re.search(r'\[MR\]:\s?(\d+\.\d+)', line)
            assert match, print(line)
            mr.append(float(match.groups()[0]))

            match = re.search(r'\[Time\]:\s?(\d+\.\d+)', line)
            if match: t.append(float(match.groups()[0]))
        else:
            pass

    assert len(endpoints) == len(ma)
    return endpoints, ma, mma, mr, t


# read file
fp16_results = extract(f"output/fp16_memory_{args.max_new_tokens}.log")
gptq4_results = extract(f"output/gptq4_memory_{args.max_new_tokens}.log")
awq_results = extract(f"output/awq_memory_{args.max_new_tokens}.log")
fp16_wo_fa_results = extract(f"output/fp16_memory_{args.max_new_tokens}_no_flash_attn.log")
assert gptq4_results[0] == fp16_results[0]
assert awq_results[0] == fp16_results[0]
assert len(fp16_wo_fa_results[0]) == len(fp16_results[0])

# memory related
df = pd.DataFrame({"endpoints": fp16_results[0]})

df["fp16_ma"] = fp16_results[1]
df["fp16_mma"] = fp16_results[2]
df["fp16_mr"] = fp16_results[3]
df["fp16_ma_deweight"] = (df["fp16_ma"] - df["fp16_ma"].values[0]).round(4)
df["fp16_mma_deweight"] = (df["fp16_mma"] - df["fp16_ma"].values[0]).round(4)
df["fp16_mr_deweight"] = (df["fp16_mr"] - df["fp16_ma"].values[0]).round(4)

df["gptq4_ma"] = gptq4_results[1]
df["gptq4_mma"] = gptq4_results[2]
df["gptq4_mr"] = gptq4_results[3]
df["gptq4_ma_deweight"] = (df["gptq4_ma"] - df["gptq4_ma"].values[0]).round(4)
df["gptq4_mma_deweight"] = (df["gptq4_mma"] - df["gptq4_ma"].values[0]).round(4)
df["gptq4_mr_deweight"] = (df["gptq4_mr"] - df["gptq4_ma"].values[0]).round(4)

df["awq_ma"] = awq_results[1]
df["awq_mma"] = awq_results[2]
df["awq_mr"] = awq_results[3]
df["awq_ma_deweight"] = (df["awq_ma"] - df["awq_ma"].values[0]).round(4)
df["awq_mma_deweight"] = (df["awq_mma"] - df["awq_ma"].values[0]).round(4)
df["awq_mr_deweight"] = (df["awq_mr"] - df["awq_ma"].values[0]).round(4)

df["fp16_wo_fa_ma"] = fp16_wo_fa_results[1]
df["fp16_wo_fa_mma"] = fp16_wo_fa_results[2]
df["fp16_wo_fa_mr"] = fp16_wo_fa_results[3]
df["fp16_wo_fa_ma_deweight"] = (df["fp16_wo_fa_ma"] - df["fp16_wo_fa_ma"].values[0]).round(4)
df["fp16_wo_fa_mma_deweight"] = (df["fp16_wo_fa_mma"] - df["fp16_wo_fa_ma"].values[0]).round(4)
df["fp16_wo_fa_mr_deweight"] = (df["fp16_wo_fa_mr"] - df["fp16_wo_fa_ma"].values[0]).round(4)

# memory gap between different models
df["gptq4_fp16_ma_gap"] = (df["gptq4_ma_deweight"] - df["fp16_ma_deweight"]).round(4)
df["awq_fp16_ma_gap"] = (df["awq_ma_deweight"] - df["fp16_ma_deweight"]).round(4)
df["fp16_wo_fa_fp16_ma_gap"] = (df["fp16_wo_fa_ma_deweight"] - df["fp16_ma_deweight"]).round(4)

# save
df.to_excel("output/qwen7b_memory.xlsx", index=False)
df_filtered = df[~df["endpoints"].str.startswith("Step")]
df_filtered.to_excel("output/qwen7b_memory_filtered.xlsx", index=False)

# elapsed time
times = pd.DataFrame({"endpoints": [f"Step {i}" for i in range(len(fp16_results[4]))]})
times["fp16"] = fp16_results[4]
times["gptq4"] = gptq4_results[4]
times["awq"] = awq_results[4]
times["fp16_wo_fa"] = fp16_wo_fa_results[4]

# save
times.to_excel("output/qwen7b_runtime_per_step.xlsx", index=False)

