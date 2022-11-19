#Make Image, Segmentation and Mask Patches
#Number of original images 1066
#    SIZE   STRIDE PATCHES DISK
#  300x300  300    4264    1.3GB
#  300x300  150    9594    2.8GB
#  200x200  100    26650   3.5GB
#  128x128  32     255840  15GB

SIZE   = 300      #Square patches of size SIZExSIZE 
STRIDE = 150      
SET = 'Resized_Datasets'  # 'Resized_Datasets' or 'Original_Datasets'
DIR = ['/Stare/images/', '/Stare/labels/','/Stare/mask/',
       '/DRIVE/test/images/','/DRIVE/test/mask/',
       '/DRIVE/training/images/','/DRIVE/training/mask/',
       '/DRIVE/training/1st_manual/',
       '/CHASE_DB1/images/','/CHASE_DB1/labels/','/CHASE_DB1/mask/']

import numpy as np
import os
from matplotlib import image
import pickle

def make_patches():
    cwd = os.getcwd()
    cwd = cwd.replace('\\','/')
    for dir in DIR:
        os.chdir(cwd + '/' + SET + dir)
        for img in os.listdir():
            p2 = cwd + '/PATCHES' + '_' + SET + '_' + str(SIZE) + '_' + str(STRIDE) + dir
            if not os.path.exists(p2): os.makedirs(p2) 
            name, _ = os.path.splitext(img)
            im = image.imread(img)
            dims = im.shape
            H = dims[0]
            W = dims[1]
            nH = int(np.floor( (H - SIZE) / STRIDE) + 1)
            nW = int(np.floor( (W - SIZE) / STRIDE) + 1)
            #We add one more dimension in the width and height to capture missed edges
            #The number of patches per image is nH*nW + nH + nW + 1
            for i in range(nH+1):
                ir = range(i*STRIDE,i*STRIDE+SIZE) if i<nH else range(H-SIZE,H)
                for j in range(nW+1):
                    jr = range(j*STRIDE,j*STRIDE+SIZE) if j<nW else range(W-SIZE,W)
                    patch = im[np.ix_(ir,jr)] if len(dims)<3 else im[np.ix_(ir,jr,range(3))]
                    patch_path = p2 + name + '_' + str(ir[0]) + '_' + str(jr[0]) + '.pickle'
                    with open(patch_path, 'wb') as f:
                        pickle.dump(patch,f)

if __name__ == '__main__':
   make_patches()
