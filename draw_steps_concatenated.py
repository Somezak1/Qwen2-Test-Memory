import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


df1 = pd.read_excel("output/qwen7b_memory.xlsx")
df2 = pd.read_excel("output/qwen7b_memory_filtered.xlsx")
endpoints = [t.split(" in ")[0] for t in df2["endpoints"].values]
ma = df2["fp16_ma_deweight"].values
mma = df2["fp16_mma_deweight"].values
mr = df2["fp16_mr_deweight"].values


def plot_between_steps(start, end):
    fig = plt.figure(figsize=(70, 8))
    if start == 0:
        plt.plot(endpoints[:4], ma[:4], 'go-')

    for i in range(start, end):
        x = endpoints[3 + 5 * i: 4 + 5 * (i + 1)]
        y_ma = ma[3 + 5 * i: 4 + 5 * (i + 1)]
        y_mma = mma[3 + 5 * i: 4 + 5 * (i + 1)]

        label_ma = f'loop {i} ma'
        label_mma = f'loop {i} mma'

        line_ma = plt.plot(x, y_ma, label=label_ma)
        line_mma = plt.plot(x, y_mma, color=line_ma[0].get_color(), linestyle='--', label=label_mma)

        for j in range(len(x)):
            try:
                dot_ma = plt.plot(x[j], y_ma[j], ".", markersize=15, color=dot_ma[0].get_color())
            except:
                dot_ma = plt.plot(x[j], y_ma[j], ".", markersize=15)
            if j == 1:
                text_ma = plt.text(x[j], y_ma[j] - 1.5, y_ma[j], color=line_ma[0].get_color())
            else:
                text_ma = plt.text(x[j], y_ma[j] + 1, y_ma[j], color=line_ma[0].get_color())

        for j in range(len(x)):
            try:
                dot_mma = plt.plot(x[j], y_mma[j], ".", markersize=15, color=dot_mma[0].get_color())
            except:
                dot_mma = plt.plot(x[j], y_mma[j], ".", markersize=15)

            text_mma = plt.text(x[j], y_mma[j] + 1, y_mma[j], color=line_ma[0].get_color())

    if start <= 200:
        plt.ylim(ymax=32, ymin=-2)
    else:
        plt.ylim(ymax=72, ymin=28)
    plt.xticks(rotation=8)
    plt.tick_params(labelsize=10)
    plt.ylabel("Memory Usage After Deducting Model Weights (MB)", size=15)
    plt.legend()
    plt.savefig(f"output/steps_concatenated_{start}_{end}.png")


plot_between_steps(0, 5)
plot_between_steps(105, 110)
plot_between_steps(413, 418)