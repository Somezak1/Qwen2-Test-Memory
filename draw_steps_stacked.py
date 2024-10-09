import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


df1 = pd.read_excel("output/qwen7b_memory.xlsx")
df2 = pd.read_excel("output/qwen7b_memory_filtered.xlsx")
endpoints = [t.split(" in ")[0] for t in df2["endpoints"].values]
ma = df2["fp16_ma_deweight"].values
mma = df2["fp16_mma_deweight"].values
mr = df2["fp16_mr_deweight"].values


fig = plt.figure(figsize=(22, 9))
end = 110
for i in range(end):
    x = [1, 2, 3, 4, 5]
    y_ma = ma[3 + 5 * i: 3 + 5 * (i + 1)]

    label_ma = f'loop {i} ma'
    label_mma = f'loop {i} mma'

    if i in [0, 1, 2, 56]:
        line_ma = plt.plot(x, y_ma, label=label_ma, linewidth=3)
        dot_ma = plt.plot(x, y_ma, ".", markersize=10)
        if i == 0:
            for j in range(len(x)):
                if j <= 2:
                    text_mma = plt.text(x[j], y_ma[j] - 0.85, y_ma[j], color=line_ma[0].get_color(), size=12)
                else:
                    text_mma = plt.text(x[j]-0.05, y_ma[j] - 0.85, y_ma[j], color=line_ma[0].get_color(), size=12)
        elif i == 1:
            for j in range(len(x)):
                if j <= 2:
                    text_mma = plt.text(x[j]-0.05, y_ma[j] + 0.5, y_ma[j], color=line_ma[0].get_color(), size=12)
                else:
                    text_mma = plt.text(x[j] - 0.05, y_ma[j] - 0.7, y_ma[j], color=line_ma[0].get_color(), size=12)
        elif i == 2:
            for j in range(len(x)):
                if j <= 2:
                    text_mma = plt.text(x[j]-0.05, y_ma[j] - 0.8, y_ma[j], color=line_ma[0].get_color(), size=12)
                else:
                    text_mma = plt.text(x[j]-0.05, y_ma[j] + 0.5, y_ma[j], color=line_ma[0].get_color(), size=12)
        elif i == 56:
            for j in range(len(x)):
                text_mma = plt.text(x[j]-0.05, y_ma[j] + 0.5, y_ma[j], color=line_ma[0].get_color(), size=12)
    elif i == end - 1:
        line_ma = plt.plot(x, y_ma, label=label_ma, linewidth=3, color=color)
        dot_ma = plt.plot(x, y_ma, ".", markersize=10)
        for j in range(len(x)):
            text_mma = plt.text(x[j], y_ma[j] + 0.3, y_ma[j], color=line_ma[0].get_color(), size=12)
    else:
        if i == 4:
            line_ma = plt.plot(x, y_ma, linewidth=0.7)
            color = line_ma[0].get_color()
        else:
            line_ma = plt.plot(x, y_ma, linewidth=0.7)


plt.ylim(ymax = 28, ymin = -2)
plt.xticks(x, [s.replace('0', 'i') for s in endpoints[3:8]])
plt.tick_params(labelsize=13)
plt.ylabel("Memory Usage After Deducting Model Weights (MB)", size=15)
plt.legend()
plt.savefig("output/steps_stacked.png")