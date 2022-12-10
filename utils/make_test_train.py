#Make the 4D test and training sets
PTRAIN = 70  #Percent of data to be used for training, the rest for testing
DIR = '/DATA_4D_Patches'

import numpy as np
import os
import shutil
from matplotlib import image
import pickle
import fnmatch
import shutil
from time import sleep

def main():
    cwd = os.getcwd()
    cwd = cwd.replace('\\','/')

    #Directors for training, validation and test sets
    p2 = cwd + DIR 
    if os.path.exists(p2): shutil.rmtree(p2) 
    p2train = p2 + '/Training'
    p2trainI = p2train + '/images/'
    p2trainL = p2train + '/labels/'
    p2val = p2 + '/Validation'
    p2valI = p2val + '/images/'
    p2valL = p2val + '/labels/'
    p2test = p2 + '/Testing'
    p2testI = p2test + '/images/'
    p2BarlowA = p2 + '/BarlowA/images/'
    p2BarlowB = p2 + '/BarlowB/images/'
    os.makedirs(p2trainI) 
    os.makedirs(p2trainL) 
    os.makedirs(p2valI) 
    os.makedirs(p2valL) 
    os.makedirs(p2testI) 
    os.makedirs(p2BarlowA) 
    os.makedirs(p2BarlowB) 


    #CHASE
    p0L = cwd + '/Resized_Datasets/CHASE_DB1/labels/'   
    p1I = cwd + '/PATCHES_Resized_Datasets_128_32_4d/CHASE_DB1/images/'  
    p1L = cwd + '/PATCHES_Resized_Datasets_128_32_4d/CHASE_DB1/labels/' 

    files0 = os.listdir(p0L)
    for item in files0.copy():
        if '1st' in item:
            files0.remove(item)
    files0 = [item.replace('_2ndHO.png','') for item in files0.copy()]
    ntrain = int(np.ceil(PTRAIN*len(files0)/100))   

    filesL = os.listdir(p1L)
    for item in filesL.copy():
        if '1st' in item:
            filesL.remove(item)

    filesI = os.listdir(p1I)
    n = -1
    for f in files0:
        n += 1
        ptargetI = p2trainI if n<ntrain else p2valI   
        ptargetL = p2trainL if n<ntrain else p2valL   
        for g in filesL:
            if f in g:
                name = g.replace('_2ndHO','')

                originalI = p1I+name
                targetI = ptargetI + 'Chase' + name
                shutil.copyfile(originalI, targetI)
                if n < ntrain:
                    shutil.copyfile(originalI, p2BarlowA + 'Chase' + name)
                    shutil.copyfile(originalI, p2BarlowB + 'Chase' + name)

                originalL = p1L+g
                targetL = ptargetL+'Chase' + name
                shutil.copyfile(originalL, targetL)
                    

    #STARE
    p0L = cwd + '/Resized_Datasets/Stare/labels/'  
    p1I = cwd + '/PATCHES_Resized_Datasets_128_32_4d/Stare/images/'  
    p1L = cwd + '/PATCHES_Resized_Datasets_128_32_4d/Stare/labels/' 
    files0 = os.listdir(p0L)

    for item in files0.copy():
        if 'ah' in item or 'vessels' in item:
            files0.remove(item)
    files0 = [item.replace('.vk','') for item in files0.copy()]
    files0 = [item.replace('.ppm','') for item in files0.copy()]
    ntrain = int(np.ceil(PTRAIN*len(files0)/100))   

    filesI = os.listdir(p1I)
    filesL = os.listdir(p1L)

    for item in filesL.copy():
        if 'ah' in item or 'vessels' in item:
            filesL.remove(item)

    n = -1
    for f in files0:
        n += 1
        ptargetI = p2trainI if n<ntrain else p2valI   
        ptargetL = p2trainL if n<ntrain else p2valL   
        for g in filesL:
            if f in g:
                name = g.replace('.vk','')

                originalI = p1I+name
                targetI = ptargetI + 'Stare' + name
                shutil.copyfile(originalI, targetI)
                if n < ntrain:
                    shutil.copyfile(originalI, p2BarlowA + 'Stare' + name)

                originalL = p1L+g
                targetL = ptargetL+'Stare' + name
                shutil.copyfile(originalL, targetL)

    del files0[0:ntrain]
    files1 = os.listdir(p1I)
    for item0 in files0:
        for item1 in files1.copy():
            if item0 in item1:
                files1.remove(item1)

    for f in files1:
        originalI = p1I+f
        targetI = p2BarlowB + 'Stare' + f
        shutil.copyfile(originalI, targetI)


    #DRIVE
    p0L = cwd + '/Resized_Datasets/Drive/training/images/'   
    p1I = cwd + '/PATCHES_Resized_Datasets_128_32_4d/Drive/training/images/'  
    p1L = cwd + '/PATCHES_Resized_Datasets_128_32_4d/Drive/training/labels/' 
    p1Itest = cwd + '/PATCHES_Resized_Datasets_128_32_4d/Drive/test/images/'  
    files0 = os.listdir(p0L)

    files0 = [item.replace('_training.tif','') for item in files0.copy()]
    ntrain = int(np.ceil(PTRAIN*len(files0)/100))   

    filesI = os.listdir(p1I)
    filesL = os.listdir(p1L)
    filesItest = os.listdir(p1Itest)

    n = -1
    for f in files0:
        n += 1
        ptargetI = p2trainI if n<ntrain else p2valI   
        ptargetL = p2trainL if n<ntrain else p2valL   
        for g in filesL:
            if '4d' + f in g:
                name = g.replace('_manual1','_training')
                originalI = p1I+name
                name = name.replace('_training','')

                targetI = ptargetI + 'Drive' + name
                shutil.copyfile(originalI, targetI)
                if n < ntrain:
                    shutil.copyfile(originalI, p2BarlowA + 'Drive' + name)
                    shutil.copyfile(originalI, p2BarlowB + 'Drive' + name)

                originalL = p1L+g
                targetL = ptargetL+'Drive' + name
                shutil.copyfile(originalL, targetL)

    for f in filesItest:
        originalI = p1Itest+f
        f = 'Drive' + f
        targetI = p2testI + f
        shutil.copyfile(originalI, targetI)

if __name__ == '__main__':
   main()

