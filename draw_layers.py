import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('fivethirtyeight')


df = pd.read_excel("output/qwen7b_memory.xlsx")


def plot_between_layers(step):
    fig = plt.figure(figsize=(30, 9))
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

    for i in range(28):
        x = [
            f"Step {step} Decoder Layer {i} After Get QKV",
            f"Step {step} Decoder Layer {i} After Reshape QKV",
            f"Step {step} Decoder Layer {i} After Get Cos/Sin",
            f"Step {step} Decoder Layer {i} After Apply RoPE",
            f"Step {step} Decoder Layer {i} After Update KV Cache",
            f"Step {step} Decoder Layer {i} After Repeat KV",
            f"Step {step} Decoder Layer {i} After Exec Attention",
            f"Step {step} Decoder Layer {i} After Output Projection",
        ]
        y = df.loc[df["endpoints"].isin(x), "fp16_ma_deweight"].values
        label = f'attention layer {i}'
        line = plt.plot(endpoints, y, label=label, linewidth=3)

    plt.xticks(endpoints, rotation=8)
    plt.tick_params(labelsize=13)
    plt.ylabel("Memory Usage After Deducting Model Weights (MB)", size=15)
    plt.title(f"Memory Usage Between Attention Layers in Step {step}", size=20)
    plt.legend()
    plt.grid()
    plt.savefig(f"output/step_{step}_layers_stacked.png")


plot_between_layers(0)
plot_between_layers(1)
plot_between_layers(2)
plot_between_layers(109)
plot_between_layers(415)