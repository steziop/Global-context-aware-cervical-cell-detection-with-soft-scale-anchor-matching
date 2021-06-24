import os


def classification_AP(classifications_file, GTs_file):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    with open(GTs_file) as f:
        gts_lines = f.readlines()
    with open(classifications_file) as f:
        classifications_lines = f.readlines()
    for gt_line in gts_lines:
        imgid = gt_line.split()[0]
        gt = gt_line.split()[1]
        for classifications_line in classifications_lines:
            if imgid == classifications_line.split()[0]:
                if gt == "1":
                    if float(classifications_line.split()[1])>=0.5:
                        TP+=1
                    else:
                        FN+=1
                else:
                    if float(classifications_line.split()[1])>=0.5:
                        FP+=1
                    else:
                        TN+=1
    print("TP:",TP)
    print("FN:",FN)
    print("FP:",FP)
    print("TN:",TN)
    total = TP + TN + FP + FN
    print("total:",total)
    acc = (TP + TN)/total
    print("acc:",acc)


def get_screen_results(detection_root, screen_file):
    filenames = os.listdir(detection_root)
    screen_file = open(screen_file,'w')
    for filename in filenames:
        image_id = filename.split('.')[0]
        flag = 0
        #print(image_id)
        with open(detection_root + filename,'r') as f1:
            lines = f1.readlines()
        for line in lines:
            label, confidence, x1, y1, x2, y2 = line.split()
            if label == 'vaginalis'or label=='monilia' or label=='dysbacteriosis':
                continue
            if float(confidence) < 0.3:
                continue
            if label!="normal":
                flag = 1
                break
        screen_file.write(image_id+" "+str(flag))
        screen_file.write('\n')
    screen_file.close()



get_screen_results("C:/Users/pcl/Desktop/Object-Detection-Metrics-master/detections_classification_xp6/", "C:/Users/pcl/Desktop/Object-Detection-Metrics-master/temptest.txt")
# 预处理目录下的 classification_GT 为仅仅以是否存在abnormal细胞和微生物作为判定的GT, 而文件gt_scr.txt是以TBS判定(不包括微生物)的GT，所以以gt_scr为准，切记！！！！！！！！！！！！！！！！！
# 切记！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# classification_AP("C:/Users/pcl/Desktop/Object-Detection-Metrics-master/det_baseline_scr.txt","C:/Users/pcl/Desktop/预处理/classification_GT.txt")
classification_AP("C:/Users/pcl/Desktop/Object-Detection-Metrics-master/temptest.txt","C:/Users/pcl/Desktop/Object-Detection-Metrics-master/gt_scr.txt")
