import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

xl = pd.ExcelFile("Results.xlsx")
ft = xl.parse("D-train SFT")

if ft["Overall Accuracy"].dtype == object:
    ft["Overall Accuracy"] = ft["Overall Accuracy"].str.replace("%","").astype(float)

models = ["Llama-2-7b", "Llama-3-8b", "Llama-3-8b-Instruct"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig.subplots_adjust(wspace=0.32)

for ax, model in zip(axes, models):
    sub = ft[ft["Model"]==model+" D-train"].sort_values("Learning Rate")
    if not sub.empty:
        ax.plot(sub["Learning Rate"], 100*sub["Overall Accuracy"],
                color="#378ADD", marker="o", linewidth=2, markersize=6)

    brow = ft[(ft["Model"]==model+" D-train") & (ft["Learning Rate"]==0.00005)]
    if not brow.empty:
        ax.axhline(100*brow["Overall Accuracy"].values[0],
                    color="#378ADD", linestyle="--", linewidth=1.5, alpha=0.6)

    lrs = sorted(ft[ft["Model"]==model+" D-train"]["Learning Rate"].unique())
    ax.set_xscale("log")
    ax.set_xticks(lrs)
    ax.set_xticklabels([f"{lr:.1e}" for lr in lrs], rotation=30, fontsize=11)
    ax.set_xlabel("Learning rate", fontsize=14)
    ax.set_ylabel("Overall accuracy (%)", fontsize=14)
    ax.set_title(model, fontsize=14)
    ax.grid(axis="y", alpha=0.2)

handles = []
handles.append(mlines.Line2D([], [], color="#378ADD", marker="o", linewidth=2,
                                 markersize=6, label="Fine-tuned over $\mathbb{D}_{\mathrm{train}}$"))
handles.append(mlines.Line2D([], [], color="#378ADD", linestyle="--",
                              linewidth=1.5, label="Baseline (fine-tuned over $\mathbb{D}_{\mathrm{train}}$ with learning rate 5e-5)"))
fig.legend(handles=handles, loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.2), fontsize=14, frameon=False)

plt.suptitle("$\mathbb{D}_{\mathrm{train}}$ fine-tuning by model: overall accuracy vs learning rate",
             fontsize=18, y=1.02)
plt.savefig("Figures/D_train_SFT_plots.pdf", bbox_inches="tight")
plt.show()