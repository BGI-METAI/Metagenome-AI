import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd
import umap

from dataset import TSVDataset

def plot_embeddings_umap(config, num_samples, figname="test_emb_umap.png", figname_all="emb_umap"):
    
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

    df_test = pd.DataFrame({"x": test_emb_2d[:, 0], "y": test_emb_2d[:, 1], "label": sel_test_lab})
    df_all = pd.DataFrame({"x": all_emb_2d[:, 0], "y": all_emb_2d[:, 1], "label": all_emb_lab})
        
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="x", y="y", data=df_test, hue="label", palette={0: 'dodgerblue', 1: 'red'}, alpha=0.8, s=25)
    plt.title("UMAP Projection of Test Data")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend(title='Protein Status', labels=['Inactive', 'Active'], loc='upper right')
    sns.despine()
    plt.savefig(figname, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="x", y="y", data=df_all, hue="label", palette={"Train": 'chartreuse', "Validation": 'deepskyblue', "Test": 'deeppink'}, alpha=0.8, s=25)
    plt.title("UMAP Projection of All Data (Train, Validation, Test)")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend(title='Dataset', loc='upper right')
    sns.despine()
    plt.savefig(figname_all, bbox_inches="tight")
    plt.close()

def select_random_sequences(dataset, num_samples):

    active, inactive = [], []

    for sample in dataset:
        if sample["labels"][0][0] == 1.0:
            active.append(sample)
        else:
            inactive.append(sample)

    if len(active) == 0: 
        num_samples_per_class = min(num_samples, len(inactive))
        rand_sel_active = []  
        rand_sel_inactive = random.sample(inactive, num_samples_per_class)
    elif len(inactive) == 0:  
        num_samples_per_class = min(num_samples, len(active))
        rand_sel_inactive = [] 
        rand_sel_active = random.sample(active, num_samples_per_class)
    else:
        num_samples_per_class = min(num_samples // 2, len(active), len(inactive))
        rand_sel_active = random.sample(active, num_samples_per_class)
        rand_sel_inactive = random.sample(inactive, num_samples_per_class)

    rand_sel = rand_sel_active + rand_sel_inactive
    random.shuffle(rand_sel)

    sel_emb, sel_lab = [], []

    for sample in rand_sel:
        sel_emb.append(sample["emb"].numpy())
        sel_lab.append(sample["labels"][0][0].item())

    return sel_emb, sel_lab