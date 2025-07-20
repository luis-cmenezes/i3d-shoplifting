import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/luis/tcc/code/checkpoints/shoplifting-training/training_log.csv")

metrics = df.columns[1:]
n = len(metrics)
cols = 2
rows = (n + 1) // cols

fig, axs = plt.subplots(rows, cols, figsize=(10, 4 * rows))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    axs[i].plot(df["epoch"], df[metric], marker="o")
    axs[i].set_title(f"{metric} x Epoch")
    axs[i].set_xlabel("Epoch")
    axs[i].set_ylabel(metric)
    axs[i].set_ylim(0, 1)
    axs[i].grid(True)

for i in range(len(metrics), len(axs)):
    fig.delaxes(axs[i])  # Remove subplots não usados

plt.tight_layout()
plt.savefig("all_metrics_vs_epoch.png")
plt.close()
