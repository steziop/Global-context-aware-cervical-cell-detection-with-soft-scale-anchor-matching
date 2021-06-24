import numpy as np
import os

def divide(rootdir):
    filenames = os.listdir(rootdir + "/Annotations")
    np.random.shuffle(filenames)
    n = len(filenames)
    trainval_split = 0.8
    train_split = 0.9
    trainval = int(n * trainval_split)
    train = int(trainval * train_split)
    i = 0
    if  not os.path.exists(rootdir + "/ImageSets"):
        os.mkdir(rootdir + "/ImageSets")
        os.mkdir(rootdir + "/ImageSets/Main")
    trainlist = filenames[:train]
    vallist = filenames[train:trainval]
    testlist = filenames[trainval:]
    with open(rootdir + "/ImageSets/Main/train.txt","a") as f:
        for image in trainlist:
            f.write(image[:-4]+"\n")
    with open(rootdir + "/ImageSets/Main/val.txt","a") as f:
        for image in vallist:
            f.write(image[:-4]+"\n")
    with open(rootdir + "/ImageSets/Main/test.txt","a") as f:
        for image in testlist:
            f.write(image[:-4]+"\n")
    
divide("/root/userfolder/dataset")