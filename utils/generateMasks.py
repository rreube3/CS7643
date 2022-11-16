import torch.cuda as cuda
from PIL import Image
import numpy as np
import os

__input_dir__: str = "../../../raw_data/"
__output_dir__: str = "mask/"


def maskData(data_set_path, thresh):
    path = __input_dir__ + data_set_path
    for filename in os.listdir(path):
        try:
            rawIm = Image.open(path + filename)
            im = np.array(rawIm)
            if len(im.shape) == 3:
                tmp = im[:, :, 0]
                tmp[tmp < thresh] = 0
                tmp[tmp > 0] = 1
                newIm = Image.fromarray(np.uint8(tmp * 255), 'L')
                newIm.save(path + __output_dir__ + "mask" + filename)
        except PermissionError as err:
            pass


if __name__ == "__main__":
    print(f"Torch is installed with cuda: {cuda.is_available()}")
    maskData("chase_db1/", 10)
    maskData("stare/all-images/", 30)
