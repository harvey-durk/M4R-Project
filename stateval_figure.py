import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

xl = pd.ExcelFile("Results.xlsx")
ft = xl.parse("StatEval SFT")
bl = xl.parse("Baseline")

for df in [ft, bl]:
    if df["Overall Accuracy"].dtype == object:
        df["Overall Accuracy"] = df["Overall Accuracy"].str.replace("%","").astype(float)

tricks = ["0-shot", "0-shot-CoT", "1-shot", "1-shot-CoT", "1-shot + DK"]
models = ["Llama-2-7b", "Llama-3-8b", "Llama-3-8b-Instruct"]
colors = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD", "#BA7517"]
markers = ["o", "s", "^", "D", "*"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig.subplots_adjust(wspace=0.32)

for ax, model in zip(axes, models):
    for trick, c, mk in zip(tricks, colors, markers):
        sub = ft[(ft["Model"]==model+" StatEval") &
                 (ft["Prompting Strategy"] == trick)].sort_values("Learning Rate")
        if not sub.empty:
            ax.plot(sub["Learning Rate"], 100*sub["Overall Accuracy"],
                    color=c, marker=mk, linewidth=2, markersize=6, label=trick)

        brow = bl[(bl["Model"]==model) &
                  (bl["Prompting Strategy"] == trick)]
        if not brow.empty:
            ax.axhline(100*brow["Overall Accuracy"].values[0],
                       color=c, linestyle="--", linewidth=1.5, alpha=0.6)

    lrs = sorted(ft["Learning Rate"].unique())
    ax.set_xscale("log")
    ax.set_xticks(lrs)
    ax.set_xticklabels([f"{lr:.1e}" for lr in lrs], rotation=30, fontsize=11)
    ax.set_xlabel("Learning rate", fontsize=14)
    ax.set_ylabel("Overall accuracy (%)", fontsize=14)
    ax.set_title(model, fontsize=14)
    ax.grid(axis="y", alpha=0.2)

handles = []
for trick, c, mk in zip(tricks, colors, markers):
    handles.append(mlines.Line2D([], [], color=c, marker=mk, linewidth=2,
                                 markersize=6, label=trick))
handles.append(mlines.Line2D([], [], color="gray", linestyle="--",
                              linewidth=1.5, label="Baseline (no fine-tune)"))
fig.legend(handles=handles, loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.2), fontsize=14, frameon=False)

plt.suptitle("mini-StatEval-foundational fine-tuning by model: overall accuracy vs learning rate",
             fontsize=18, y=1.02)
plt.savefig("Figures/StatEval_SFT_plots.pdf", bbox_inches="tight")
plt.show()