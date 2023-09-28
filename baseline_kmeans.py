import os 
os.environ['OMP_NUM_THREADS'] = '50' 
from utils import compute_metrics
from model_builders import load_embeds, available_models
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import pickle

import model_builders
from gen_embeds import main as gen_embeds

def load_data(args):
    dataset = args.dataset
    arch = args.arch
    try:
        emb, targets_train = load_embeds(arch=arch, dataset=dataset, test=False, with_label=True)
        emb_test, targets_test = load_embeds(arch=arch, dataset=dataset, test=True, with_label=True)
    except:
        print("Generating embeds for ", arch, dataset)
        gen_embeds(args)
        emb, targets_train = load_embeds(arch=arch, dataset=dataset, test=False, with_label=True)
        emb_test, targets_test = load_embeds(arch=arch, dataset=dataset, test=True, with_label=True)
    if args.normalize:
        mean, std = emb.mean(dim=0), emb.std(dim=0)
        emb = (emb - mean) / std
        emb_test = (emb_test - mean) / std
    return emb, targets_train, emb_test, targets_test


def kmeans_baseline(args, dir_path, num=1):
    emb, targets_train, emb_test, targets_test = load_data(args)
    
    rows = []
    n_clusters = {
        'CIFAR100': 100,
        'CIFAR10': 10,
        'STL10': 10,
        'IN1K': 1000,
        'IN100': 100,
        'IN50': 50,
        'IN200': 200,
    }[args.dataset]
    for _ in tqdm(range(num), leave=False):
        arch_name = args.arch.replace("/", "_")
        kmeans_save_path = dir_path / f"{arch_name}_kmeans.pkl"
        if not os.path.exists(kmeans_save_path):
            # fit based on train set
            print('Fitting K-Means classifier..')
            kmeans = KMeans(n_clusters=n_clusters, n_init=2, verbose=True).fit(emb)
            # save kmeans model
            print('Saving K-Means classifier..')
            pickle.dump(kmeans, open(kmeans_save_path, "wb"))
        else:
            print("kmeans model already exists.")
            kmeans = pickle.load(open(kmeans_save_path, "rb"))

        preds = torch.tensor(kmeans.predict(emb_test))
        data = compute_metrics(targets_test, preds)
        data = dict(zip(["Accuracy", "NMI", "Adjusted NMI", "Adjusted Rand-Index"], data))
        rows.append(data)
    return pd.DataFrame(rows)


def true_label_means_baseline(args, num=1):
    emb, targets_train, emb_test, targets_test = load_data(args)
    rows = []
    n_clusters = {
        'CIFAR100': 100,
        'CIFAR10': 10,
        'STL10': 10,
        'IN1K': 1000,
        'IN100': 100,
        'IN50': 50,
        'IN200': 200,
    }[args.dataset]
    centers = torch.stack([emb[targets_train == i].mean(dim=0) for i in range(n_clusters)])
    for _ in tqdm(range(num), leave=False):
        preds = torch.cdist(centers, emb_test).argmin(dim=0)
        data = compute_metrics(targets_test, preds)
        data = dict(zip(["Accuracy", "NMI", "Adjusted NMI", "Adjusted Rand-Index"], data))
        rows.append(data)
    return pd.DataFrame(rows)


def agg_stats(df, arch):
    name = pd.DataFrame({"arch":[arch]})
    means = df.mean().round(2).to_frame().T
    means.rename(columns = {'Accuracy':'Mean Acc', "NMI":"Mean NMI","Adjusted NMI":"Mean ANMI","Adjusted Rand-Index":"Mean ARI"}, inplace = True)
    stds = df.std().round(2).to_frame().T
    stds.rename(columns = {'Accuracy':'Std Acc', "NMI":"Std NMI","Adjusted NMI":"Std ANMI","Adjusted Rand-Index":"Std ARI"}, inplace = True)
    return pd.concat([name, means, stds], axis=1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['CIFAR100', 'CIFAR10', "STL10", \
                                                                "CIFAR20", "IN1K", "IN50", 'IN100', "IN200", "IN1K"], type=str)
    parser.add_argument('--archs', nargs='+', default=available_models())
    parser.add_argument('--outpath', type=Path, default=Path('data'))
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--datapath', default='./data', type=str)
    parser.add_argument('--normalize', action='store_true', default=False)
    args = parser.parse_args()
    file_prefix = 'kmeans_normalized' if args.normalize else 'kmeans'
    dsets = [args.dataset]
    list_df_kmeans = []
    list_df_true_means = []
    for arch in tqdm(args.archs):
        arch_name = arch.replace("/", "_")
        for dset in dsets:
            dir_path = Path(f"experiments/clustering/kmeans_baseline/{dset}").expanduser().resolve()
            dir_path.mkdir(parents=True, exist_ok=True)
            args.arch = arch
            args.dataset = dset
            if os.path.isfile(dir_path / f"{file_prefix}_{arch_name}_{dset}.csv") and os.path.isfile(dir_path / f"labeled_centers_{arch_name}_{dset}.csv"):
                list_df_kmeans.append(pd.read_csv(dir_path / f"{file_prefix}_{arch_name}_{dset}.csv"))
                list_df_true_means.append(pd.read_csv(dir_path / f"{file_prefix}_{arch_name}_{dset}.csv"))
                continue
            res_kmeans = kmeans_baseline(args, dir_path, num=1)
            agg_df = agg_stats(res_kmeans, arch)
            list_df_kmeans.append(agg_df)
            agg_df.to_csv(dir_path / f"{file_prefix}_{arch_name}_{dset}.csv")

            agg_true_means = true_label_means_baseline(args, num=1)
            agg_df_true_means = agg_stats(agg_true_means, arch)
            list_df_true_means.append(agg_df_true_means)
            agg_df_true_means.to_csv(dir_path / f"labeled_centers_{arch_name}_{dset}.csv")
        
    df_all_km = pd.concat(list_df_kmeans, axis=0, ignore_index=True)
    df_all_km.to_csv(dir_path / f"{file_prefix}_ALL.csv")

    df_all_tm = pd.concat(list_df_true_means, axis=0, ignore_index=True )
    df_all_tm.to_csv(dir_path / f"labeled_centers_ALL.csv")
        
