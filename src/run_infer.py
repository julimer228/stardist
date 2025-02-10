import os
from pathlib import Path
import argparse
import cv2
import numpy as np
from stardist.models import StarDist2D
from tqdm import tqdm
import warnings

from src.utils.image_utils import relabel_image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", message=".*iCCP.*")

"""
Author: Julia Merta
Date: 02-01-2025

Info: 

      You can run this script from the command line (read the parameters info below).
      Another way is to set the correct paths and options here in the code and run.
      
      Script to run the inference on the images from the test dataset. Predictions
      will be saved in the chosen folder. Both .png masks and numpy arrays masks 

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Divide data into patches")
    parser.add_argument(
        '--images',
        type=str,
        default="F:/Cell Detection Visual Data/Data/BCCD Dataset with mask/test/original/",
        help='Folder with images'
    )
    parser.add_argument(
        '--masks',
        type=str,
        default="F:/Cell Detection Visual Data/Data/BCCD Dataset with mask/test/mask/",
        help='Folder with images'
    )
    parser.add_argument(
        '--res_folder',
        type=str,
        default="F:/Cell Detection Visual Data/Data/BCCD Dataset with mask/test/pred/",
        help='Folder for predictions'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default="E:/stardist/src/base-model-no-aug/",
        help='Path with the weights of the model (file weights_best.h5 will be used for inference)'
    )

    args = parser.parse_args()

    res_numpy_pred = os.path.join(args.res_folder, "numpy_pred")
    res_numpy_gt = os.path.join(args.res_folder, "numpy_gt")
    res_png_pred = os.path.join(args.res_folder, "res_png_pred")

    os.makedirs(res_numpy_pred, exist_ok=True)
    os.makedirs(res_numpy_gt, exist_ok=True)
    os.makedirs(res_png_pred, exist_ok=True)

    img_paths = sorted(Path(args.images).glob('*.*'))
    masks_paths = sorted(Path(args.masks).glob('*.*'))

    X = list(map(lambda path: cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), img_paths))
    Y_gt = list(map(lambda path: cv2.imread(path, cv2.IMREAD_GRAYSCALE), masks_paths))
    Y_gt = [relabel_image(y) for y in Y_gt]
    X = [x / 255.0 for x in X]  # normalize

    model = StarDist2D(None, name='stardist', basedir=args.model_path)

    Y_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X)]

    file_names = os.listdir(args.images)

    for y, y_gt, path in tqdm(zip(Y_pred, Y_gt, file_names), total=len(file_names), desc="Processing"):
        basename = path.split(".")[0] + ".npy"
        np.save(os.path.join(res_numpy_pred, basename), y)
        np.save(os.path.join(res_numpy_gt, basename), y_gt)
        cv2.imwrite(os.path.join(res_png_pred, path), y)
