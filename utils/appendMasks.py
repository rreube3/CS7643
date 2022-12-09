from typing import List, Callable
import os
import numpy as np
import pickle
from PIL import Image

path_to_patches: str = "A:/DATA_4D_Patches/DATA_4D_Patches/Testing/test-20221209T003851Z-001/"
subdirs: List[str] = ["test/"]


def appendMasks(ds_name: str, labels_dir_name: str, mask_path_fn: Callable, use_subdirs: bool = False) -> None:
    """
    Adds mask as 4th channel.

    :param ds_name: the data set name
    :param labels_dir_name: the name of the labels directory
    :param mask_path_fn: the function for mapping (path, img name) -> mask name
    :param use_subdirs: whether the dataset has subdirs, if True uses the subdirs list
    :return: None
    """
    paths: List[str] = [path_to_patches + ds_name]
    if use_subdirs:
        tmp: List[str] = []
        for i in range(len(subdirs)):
            tmp.append(path_to_patches + ds_name + subdirs[i])
        paths = tmp

    for path in paths:
        img_path = path + "images/"
        label_path = path + labels_dir_name
        saveNewImgs(img_path, path, mask_path_fn)
        # if not ((ds_name == "DRIVE/") and ("test" in path)):
        #     saveNewImgs(label_path, path, mask_path_fn)


def saveNewImgs(image_path, path, mask_path_fn):
    for filename in os.listdir(image_path):
        if "4d" != filename[:2]:
            try:
                img_arr = np.array(Image.open(image_path + filename))
                mask_file_name = labelNameToImgName(filename)
                mask_path = mask_path_fn(path, mask_file_name)
                mask_arr = np.array(Image.open(mask_path))
                if len(mask_arr.shape) == 3:
                    mask_arr = mask_arr[:, :, 0]
                if len(img_arr.shape) == 3:
                    img_arr_split = [np.squeeze(x, axis=2) if len(x.shape) == 3 else x for x in np.array_split(img_arr, 3, axis=2)]
                else:
                    img_arr_split = [img_arr for i in range(3)]
                img_arr_split.append(mask_arr)
                new_arr = np.stack(img_arr_split, axis=2)
                with open(image_path + "4d" + filename.replace("tif", "pickle"), 'wb') as f:
                    pickle.dump(new_arr, f)
            except PermissionError as err:
                pass


def labelNameToImgName(label_name: str) -> str:
    """
    Masks are named from the training image the label matches,
    .so this function halps resolve the label image to the training image

    :param label_name: label file name
    :return: the respective training image filename.
    """
    label_name = label_name.replace("_1stHO", "")
    label_name = label_name.replace("_2ndHO", "")
    label_name = label_name.replace(".ah", "")
    label_name = label_name.replace(".vk", "")
    label_name = label_name.replace("-vessels4", "")
    label_name = label_name.replace("manual1", "training")
    return label_name


def drive_mask_path_fn(path, filename):
    nm_arr = filename.split('.')
    nm_arr[0] = nm_arr[0] + "_mask"
    nm_arr[1] = "gif"
    # add back splitter
    for i in range(1, len(nm_arr)):
        nm_arr[i] = '.' + nm_arr[i]
    res = path + "mask/" + ''.join(nm_arr)
    return res


if __name__ == "__main__":
    # appendMasks("CHASE_DB1/", "labels/", lambda p, f: p + "mask/mask" + f)
    appendMasks("", "1st_manual/", drive_mask_path_fn, use_subdirs=True)
    # appendMasks("Stare/", "labels/", lambda p, f: p + "mask/mask" + f)
