import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

xl = pd.ExcelFile("Results.xlsx")
ft = xl.parse("D-train + StatEval SFT")
bl = xl.parse("D-train SFT")

for df in [ft, bl]:
    if df["Overall Accuracy"].dtype == object:
        df["Overall Accuracy"] = df["Overall Accuracy"].str.replace("%","").astype(float)

models = ["Llama-2-7b", "Llama-3-8b", "Llama-3-8b-Instruct"]
learning_rates = [0.00089, 0.00028, 0.00028]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig.subplots_adjust(wspace=0.32)

for ax, model, learning_rate in zip(axes, models, learning_rates):
    sub = ft[ft["Model"]==model+" D-train+StatEval"].sort_values("Learning Rate")
    if not sub.empty:
        ax.plot(sub["Learning Rate"], 100*sub["Overall Accuracy"],
                color="#378ADD", marker="o", linewidth=2, markersize=6)

    brow = bl[(bl["Model"]==model+" D-train") & (bl["Learning Rate"]==learning_rate)]
    if not brow.empty:
        ax.axhline(100*brow["Overall Accuracy"].values[0],
                    color="#378ADD", linestyle="--", linewidth=1.5, alpha=0.6)

    lrs = sorted(ft["Learning Rate"].unique())
    ax.set_xscale("log")
    ax.set_xticks(lrs)
    ax.set_xticklabels([f"{lr:.1e}" for lr in lrs], rotation=30, fontsize=11)
    ax.set_xlabel("Learning rate", fontsize=14)
    ax.set_ylabel("Overall accuracy (%)", fontsize=14)
    ax.set_title(model, fontsize=14)
    ax.grid(axis="y", alpha=0.2)

handles = []
handles.append(mlines.Line2D([], [], color="#378ADD", marker="o", linewidth=2,
                                 markersize=6, label="Fine-tuned over $\mathbb{D}_{\mathrm{train}}$ and mini-StatEval-foundational"))
handles.append(mlines.Line2D([], [], color="#378ADD", linestyle="--",
                              linewidth=1.5, label="Baseline (fine-tuned over $\mathbb{D}_{\mathrm{train}}$)"))
fig.legend(handles=handles, loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.2), fontsize=14, frameon=False)

plt.suptitle("$\mathbb{D}_{\mathrm{train}}$ then mini-StatEval-foundational fine-tuning by model: overall accuracy vs learning rate",
             fontsize=18, y=1.02)
plt.savefig("Figures/D-train_StatEval_SFT_plots.pdf", bbox_inches="tight")
plt.show()