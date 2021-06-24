def get_GT(detections_GT_file, classification_GT_file):
    with open(detections_GT_file) as f:
        imglines = f.readlines()
    classification_GT_file = open(classification_GT_file,'w+')
    for imgline in imglines:
        imgpath = imgline.split()[0]
        image_id = imgpath.split('/')[-1].split('.')[0]
        imgclassification = 0
        imgboxes = imgline.split()[1:]
        for imgbox in imgboxes:
            boxcls = imgbox.split(",")[-1]
            if int(boxcls)!=0:
                imgclassification = 1
                break
        classification_GT_file.write(image_id+" "+str(imgclassification))
        classification_GT_file.write('\n')
    classification_GT_file.close()



get_GT("2019_test.txt", "classification_GT.txt")