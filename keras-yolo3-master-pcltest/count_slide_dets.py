import os
import numpy as np

rootdir ='/home/data/TCT_data100/testdata/001500002/TCT/detections_4k_nms/'
filenames = os.listdir(rootdir)


id2name = ["normal", "ascus", "asch", "lsil", "hsil", "agc", "adenocarcinoma", "vaginalis", "monilia", "dysbacteriosis"]
count = np.zeros(10)

for filename in filenames:
    detfile=rootdir + filename
    dets=np.loadtxt(detfile,dtype=bytes).astype(str)
    if dets.size == 6:
        dets = np.expand_dims(dets,0)
    cells = dets[:,0]
    for i in range(10):
        count[i] += sum(np.char.count(cells,id2name[i]))

for i in range(10):
    print(id2name[i] + ": " + str(count[i]))

    