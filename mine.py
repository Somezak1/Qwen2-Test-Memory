import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

df = pd.read_excel("output/qwen7b_memory.xlsx")
results = []

for step in [0, 1, 2, 3, 4, 109, 415]:
    endpoints = [
        f"Step {step} Decoder Layer i After Get QKV",
        f"Step {step} Decoder Layer i After Reshape QKV",
        f"Step {step} Decoder Layer i After Get Cos/Sin",
        f"Step {step} Decoder Layer i After Apply RoPE",
        f"Step {step} Decoder Layer i After Update KV Cache",
        f"Step {step} Decoder Layer i After Repeat KV",
        f"Step {step} Decoder Layer i After Exec Attention",
        f"Step {step} Decoder Layer i After Output Projection",
    ]
    y_list = []

    for layer in range(28):
        x = [endpoint.replace(" i ", f" {layer} ") for endpoint in endpoints]
        y = df.loc[df["endpoints"].isin(x), "fp16_ma_deweight"].values
        y_list.append(y)

    step_diff = np.diff(np.array(y_list), axis=0)
    assert step_diff.shape == (27, 8), print(step_diff.shape)
    step_diff_mean = np.mean(step_diff, axis=1)
    results.append(step_diff_mean)


print(np.array(results))