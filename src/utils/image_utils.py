import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from stardist import random_label_cmap


def relabel_image(image):
    label_img = label(image)
    regions = regionprops(label_img)
    output_image = np.zeros_like(label_img, dtype=np.uint8)

    for region in regions:
        if region.area >= 20:
            output_image[label_img == region.label] = region.label
    return output_image


def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai, al) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=(1.25, 1)))
    im = ai.imshow(img, cmap='gray', clim=(0, 1))
    ai.set_title(img_title)
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=random_label_cmap())
    al.set_title(lbl_title)
    plt.tight_layout()
    plt.savefig(lbl_title + ".png")

