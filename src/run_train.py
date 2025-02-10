from __future__ import print_function, unicode_literals, absolute_import, division

import random

import matplotlib
from matplotlib import pyplot as plt

from src.utils.aug import augmenter
from src.utils.image_utils import plot_img_label
from stardist import fill_label_holes, calculate_extents
from stardist.matching import matching_dataset
import os
from glob import glob
from pathlib import Path
import argparse
import cv2
import numpy as np
from stardist.models import Config2D, StarDist2D
from tqdm import tqdm
import warnings

matplotlib.rcParams["image.interpolation"] = 'none'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings("ignore", message=".*iCCP.*")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train stardist model")
    parser.add_argument(
        '--images',
        type=str,
        default="F:/Cell Detection Visual Data/Data/BCCD Dataset with mask/train/patches/images/*.png",
        help='Folder with images'
    )
    parser.add_argument(
        '--masks',
        type=str,
        default="F:/Cell Detection Visual Data/Data/BCCD Dataset with mask/train/patches/masks/*.png",
        help='Folder with masks'
    )
    parser.add_argument(
        '--checkpoints_dir',
        type=str,
        default="base-model-no-aug",
        help='Directory for checkpoints'
    )

    args = parser.parse_args()
    random.seed(42)
    X_paths = sorted(glob(args.images))
    Y_paths = sorted(glob(args.masks))
    indices = random.sample(range(len(X_paths)), 1000)
    X_paths = [X_paths[i] for i in indices]
    Y_paths = [Y_paths[i] for i in indices]
    assert all(Path(x).name == Path(y).name for x, y in zip(X_paths, Y_paths))

    X = list(map(lambda path: cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), X_paths))
    Y = list(map(lambda path: cv2.imread(path, cv2.IMREAD_GRAYSCALE), Y_paths))
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

    Y = [fill_label_holes(y) for y in tqdm(Y)]

    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    X_val = [x/255.0 for x in X_val]

    conf = Config2D(
        n_rays=64,
        grid=(2, 2),
        use_gpu=False,
        n_channel_in=n_channel,
        train_reduce_lr = {"factor": 0.1, "patience": 20, "min_delta": 0.01},
        train_learning_rate = 0.0003,
        train_epochs = 100
    )

    print(conf)
    vars(conf)

    model = StarDist2D(conf, name='stardist', basedir=args.checkpoints_dir)
    median_size = calculate_extents(list(Y), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")


    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter)
    #model = StarDist2D(None, name='stardist', basedir=args.checkpoints_dir)

    model.optimize_thresholds(X_val, Y_val)
    Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
                  for x in tqdm(X_val)]

    idx=3
    plot_img_label(X_val[idx], Y_val[idx], lbl_title="label GT"+str(idx))
    plot_img_label(X_val[idx], Y_val_pred[idx], lbl_title="label Pred"+str(idx))

    idx = 9
    plot_img_label(X_val[idx], Y_val[idx], lbl_title="label GT" + str(idx))
    plot_img_label(X_val[idx], Y_val_pred[idx], lbl_title="label Pred" + str(idx))

    idx = 51
    plot_img_label(X_val[idx], Y_val[idx], lbl_title="label GT" + str(idx))
    plot_img_label(X_val[idx], Y_val_pred[idx], lbl_title="label Pred" + str(idx))

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
    stats[taus.index(0.5)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
        ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax1.set_xlabel(r'IoU threshold $\tau$')
    ax1.set_ylabel('Metric value')
    ax1.grid()
    ax1.legend()

    for m in ('fp', 'tp', 'fn'):
        ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax2.set_xlabel(r'IoU threshold $\tau$')
    ax2.set_ylabel('Number #')
    ax2.grid()
    ax2.legend();
    plt.savefig("res.png")


