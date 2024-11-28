"""
Usage:
1. Prepare a `config` dictionary containing paths to datasets and parameters:
   Example:
   config = {
       "train": "path/to/train.tsv",
       "valid": "path/to/valid.tsv",
       "test": "path/to/test.tsv",
       "emb_dir": "path/to/embeddings",
       "num_samples": 4000  # Optional: Number of samples to randomly select per dataset
   }

2. Call the function:
   plot_embeddings_umap(config)

3. The function will:
   - Visualize UMAP projections for the test data (left) and all data (right).
   - Save the plot as a PNG file named based on the test dataset file.
"""

import matplotlib.pyplot as plt
import random
import seaborn as sns
import umap
import os
import logging

from sklearn.decomposition import PCA
from dataset import TSVDataset


def plot_embeddings_umap(config):

    umap_num_samples = config.get('umap_num_samples', None)

    # Skip UMAP processing if it's not defined or set to 0
    if umap_num_samples is None or umap_num_samples == 0:
        logging.info("Skipping UMAP visualization.")
        return  

    logging.info(f"Running UMAP visualization with {umap_num_samples} samples.")
    
    # Load datasets
    train_ds = TSVDataset(config["train"], config["emb_dir"], "mean")
    valid_ds = TSVDataset(config["valid"], config["emb_dir"], "mean")
    test_ds = TSVDataset(config["test"], config["emb_dir"], "mean")

    # Select random samples from datasets
    sel_train_emb, _ = select_random_sequences(train_ds, umap_num_samples)
    sel_valid_emb, _ = select_random_sequences(valid_ds, umap_num_samples)
    sel_test_emb, sel_test_lab = select_random_sequences(test_ds, umap_num_samples)

    # Combine all embeddings and labels
    all_emb = sel_train_emb + sel_valid_emb + sel_test_emb
    all_emb_lab = ["Train"] * len(sel_train_emb) + ["Validation"] * len(sel_valid_emb) + ["Test"] * len(sel_test_emb)
    
    # Apply PCA to reduce features
    pca = PCA(n_components=50)  
    all_emb_pca = pca.fit_transform(all_emb)

    # Apply UMAP to PCA-reduced embeddings
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42, n_neighbors=15)
    all_emb_pca_2d = reducer.fit_transform(all_emb_pca)
    test_emb_pca_2d = all_emb_pca_2d[-len(sel_test_emb):]
    
    # Plot UMAP projections
    figname = f"{os.path.splitext(os.path.basename(config['test']))[0]}_umap.png"
    plt.figure(figsize=(20, 8))

    # Test data projection
    ax1 = plt.subplot(1, 2, 1)
    sns.scatterplot(x=test_emb_pca_2d[:, 0], y=test_emb_pca_2d[:, 1], hue=sel_test_lab, palette={0: 'dodgerblue', 1: 'red'},
                    alpha=0.8, s=25, ax=ax1)
    ax1.set(title="UMAP Projection of Test Data", xlabel="UMAP Component 1", ylabel="UMAP Component 2")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['Inactive', 'Active'], title='Protein Status', loc='upper left')

    # All data projection
    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(x=all_emb_pca_2d[:, 0], y=all_emb_pca_2d[:, 1], hue=all_emb_lab, palette={"Train": 'chartreuse', "Validation": 'deepskyblue', "Test": 'deeppink'},
                    alpha=0.8, s=25, ax=ax2)
    ax2.set(title="UMAP Projection of All Data", xlabel="UMAP Component 1", ylabel="UMAP Component 2")
    ax2.legend(title='Dataset', loc='upper left')

    sns.despine()

    # Saving figure
    plt.tight_layout()
    plt.savefig(figname, bbox_inches="tight")
    plt.close()

    logging.info(f"UMAP visualization saved as {os.path.abspath(figname)}")


def select_random_sequences(dataset, umap_num_samples):

    active, inactive = [], []

    # Split sequences into active and inactive
    for sample in dataset:
        if sample["labels"][0][0] == 1.0:
            active.append(sample)
        else:
            inactive.append(sample)

    # Determine number of samples per class
    if len(active)==0 or len(inactive)==0:  
        num_samples_per_class = min(umap_num_samples, len(active) + len(inactive))
    else:
        num_samples_per_class = min(umap_num_samples // 2, len(active), len(inactive))

    # Randomly select samples
    rand_sel_active = [] if len(active)==0 else random.sample(active, num_samples_per_class)
    rand_sel_inactive = [] if len(inactive)==0 else random.sample(inactive, num_samples_per_class)

    rand_sel = rand_sel_active + rand_sel_inactive
    random.shuffle(rand_sel)

    # Extract embeddings and labels
    sel_emb, sel_lab = [], []

    for sample in rand_sel:
        sel_emb.append(sample["emb"].numpy())
        sel_lab.append(sample["labels"][0][0].item())

    return sel_emb, sel_lab