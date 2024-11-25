import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd
import umap
import os

from dataset import TSVDataset

def plot_embeddings_umap(config):

    num_samples = config.get('num_samples', 2000)
    
    train_ds = TSVDataset(config["train"], config["emb_dir"], "mean")
    valid_ds = TSVDataset(config["valid"], config["emb_dir"], "mean")
    test_ds = TSVDataset(config["test"], config["emb_dir"], "mean")

    sel_train_emb, _ = select_random_sequences(train_ds, num_samples)
    sel_valid_emb, _ = select_random_sequences(valid_ds, num_samples)
    sel_test_emb, sel_test_lab = select_random_sequences(test_ds, num_samples)

    all_emb = sel_train_emb + sel_valid_emb + sel_test_emb
    all_emb_lab = ["Train"] * len(sel_train_emb) + ["Validation"] * len(sel_valid_emb) + ["Test"] * len(sel_test_emb)
    
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42, n_neighbors=50)

    all_emb_2d = reducer.fit_transform(all_emb)
    test_emb_2d = all_emb_2d[-len(sel_test_emb):]

    figname = f"{os.path.splitext(os.path.basename(config['test']))[0]}_umap.png"

    plt.figure(figsize=(20, 8))

    ax1 = plt.subplot(1, 2, 1)  
    sns.scatterplot(x=test_emb_2d[:, 0], y=test_emb_2d[:, 1], hue=sel_test_lab, palette={0: 'dodgerblue', 1: 'red'},
                    alpha=0.8, s=25, ax=ax1)
    ax1.set(title="UMAP Projection of Test Data", xlabel="UMAP Component 1", ylabel="UMAP Component 2")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['Inactive', 'Active'], title='Protein Status', loc='upper left')

    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(x=all_emb_2d[:, 0], y=all_emb_2d[:, 1], hue=all_emb_lab, palette={"Train": 'chartreuse', "Validation": 'deepskyblue', "Test": 'deeppink'},
                    alpha=0.8, s=25, ax=ax2)
    ax2.set(title="UMAP Projection of All Data", xlabel="UMAP Component 1", ylabel="UMAP Component 2")
    ax2.legend(title='Dataset', loc='upper left')

    sns.despine()

    plt.tight_layout()
    plt.savefig(figname, bbox_inches="tight")
    plt.close()


def select_random_sequences(dataset, num_samples):

    active, inactive = [], []

    for sample in dataset:
        if sample["labels"][0][0] == 1.0:
            active.append(sample)
        else:
            inactive.append(sample)

    if not active or not inactive:  
        num_samples_per_class = min(num_samples, len(active) + len(inactive))
    else:
        num_samples_per_class = min(num_samples // 2, len(active), len(inactive))

    rand_sel_active = [] if not active else random.sample(active, num_samples_per_class)
    rand_sel_inactive = [] if not inactive else random.sample(inactive, num_samples_per_class)

    rand_sel = rand_sel_active + rand_sel_inactive
    random.shuffle(rand_sel)

    sel_emb, sel_lab = [], []

    for sample in rand_sel:
        sel_emb.append(sample["emb"].numpy())
        sel_lab.append(sample["labels"][0][0].item())

    return sel_emb, sel_lab