#Create Image, Segmentation and Mask Patches
SIZE = 128
STRIDE = 32
ACTION = 'MAKE'   #'SPECS' or 'MAKE'
#ACTION = 'SPECS'
DIR = ['/Stare/images/', '/Stare/labels/','/Stare/mask/',
       '/DRIVE/test/images/','/DRIVE/test/mask/',
       '/DRIVE/training/images/','/DRIVE/training/mask/',
       '/DRIVE/training/1st_manual/',
       '/CHASE_DB1/images/','/CHASE_DB1/labels/','/CHASE_DB1/mask/']
#SET = ['Original_Datasets','Resized_Datasets']
SET = ['Resized_Datasets']

import numpy as np
from PIL import Image
import os
from matplotlib import image
import pickle

def main():
  cwd = os.getcwd()
  cwd = cwd.replace('\\','/')
  if ACTION == 'MAKE':
    for set in SET:
      for dir in DIR:
        os.chdir(cwd + '/' + set + dir)
        for img in os.listdir():
            p2 = cwd + '/PATCHES' + '_' + set + '_' + str(SIZE) + '_' + str(STRIDE) + dir
            if not os.path.exists(p2):
                os.makedirs(p2)
            name, _ = os.path.splitext(img)
            im = image.imread(img)
            dims = im.shape
            h = dims[0]
            w = dims[1]
            nh = int(np.floor( (h - SIZE) / STRIDE) + 1)
            nw = int(np.floor( (w - SIZE) / STRIDE) + 1)
            oh = h - (nh-1)*STRIDE-SIZE
            ow = w - (nw-1)*STRIDE-SIZE
            offsets = [[0,0],[0,ow],[oh,0],[oh,ow]]
            for k in range(4):
              o = offsets[k]
              for i in range(nh):
                for j in range(nw):
                    ir = range(i*STRIDE+o[0],i*STRIDE+SIZE+o[0])
                    jr = range(j*STRIDE+o[1],j*STRIDE+SIZE+o[1])
                    if len(dims) == 3:  
                        patch = im[ir,jr, :]
                    else:  
                        patch = im[ir,jr]
                    patch_path = p2 + name + '_' + str(ir[0]) + '_' + str(jr[0]) + '.pickle'
                    with open(patch_path, 'wb') as f:
                        pickle.dump(patch,f)
            #break   #Use this break to patch just one image per set

  elif ACTION == 'SPECS':
    for set in SET:
      print(set)
      num_images = 0
      num_patches = 0
      for dir in DIR:
        os.chdir(cwd + '/' + set + dir)
        num_dir_patches = 0
        num_dir_original = 0

        for img in os.listdir():
            p2 = cwd + '/PATCHES' + '_' + set + '_' + str(SIZE) + '_' + str(STRIDE) + dir
            im = image.imread(img)
            dims = im.shape
            h = dims[0]
            w = dims[1]
            #print('    Image Size: ',h,w)
            nh = int(np.floor( (h - SIZE) / STRIDE) + 1)
            nw = int(np.floor( (w - SIZE) / STRIDE) + 1)
            break    #Just want the size of one image in the directory

        for img in os.listdir():
            num_dir_patches += nw*nh
            num_images += 1
            num_dir_original += 1
            num_patches += 4*nw*nh
        print('    ' + dir + '  Number of Images: ',num_dir_original,'  Number of Patches: ',num_dir_patches)
      print('    Total number of images: ',num_images)
      print('    Total number of patches: ',num_patches)

if __name__ == '__main__':
    main()
