# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:58:49 2019

@author: leechee
"""

from voc_eval_aschsil import voc_eval
import os
FindPath = 'C:/Users/pcl/Desktop/Object-Detection-Metrics-master/keras_category_results_final4000nms/'
FileNames = os.listdir(FindPath)
#for file_name in FileNames:
for file in FileNames:
    file_name = file.split('.')[0]
    #print (voc_eval('/root/userfolder/darknet/results/{}.txt','/root/userfolder/darknet/TCT/VOC2019/Annotations_4000s/{}.xml','/root/userfolder/darknet/TCT/VOC2019/writetxt/test.txt', file_name, '/root/userfolder/darknet/AP/'+file_name))
    rec, prec, ap=voc_eval('/root/userfolder/Object-Detection-Metrics/keras_category_results/{}.txt','/root/userfolder/keras-yolo3-master/VOC2019/Annotations_4000s/{}.xml','/root/userfolder/keras-yolo3-master/VOC2019/writetxt/test.txt', file_name, '/root/userfolder/Object-Detection-Metrics/AP/'+file_name)
    #print ('rec' ,rec)
    #print ('prec',prec)
    print ('ap',ap)
