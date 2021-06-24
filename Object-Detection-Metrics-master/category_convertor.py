import os
'''
with open('/root/userfolder/keras-yolo3-master/model_data/my_classes.txt') as fclass:
    classnames = fclass.read().rstrip().split()
'''


rootdir = 'C:/Users/pcl/Desktop/Object-Detection-Metrics-master/detections_final4000nms/'
filenames = os.listdir(rootdir)
for filename in filenames:
    image_id = filename.split('.')[0]
    with open(rootdir+filename,'r') as f1:
        lines = f1.readlines()
    for line in lines:
        label, confidence, x1, y1, x2, y2 = line.split()
        if image_id[0:1] == 'l':
            x1 = float(x1)*2.5
            y1 = float(y1)*2.5
            x2 = float(x2)*2.5
            y2 = float(y2)*2.5
            if image_id[8:] == '00':
                x1 = str(x1)
                y1 = str(y1)
                x2 = str(x2)
                y2 = str(y2)
            elif image_id[8:] == '01':
                x1 = str(x1)
                y1 = str(y1+1500)
                x2 = str(x2)
                y2 = str(y2+1500)
            elif image_id[8:] == '02':
                x1 = str(x1+2000)
                y1 = str(y1)
                x2 = str(x2+2000)
                y2 = str(y2)
            elif image_id[8:] == '03':
                x1 = str(x1+2000)
                y1 = str(y1+1500)
                x2 = str(x2+2000)
                y2 = str(y2+1500)
            new_image_id = image_id[1:7]
        elif len(image_id) < 8:
            x1 = float(x1)*5
            y1 = float(y1)*5
            x2 = float(x2)*5
            y2 = float(y2)*5
            x1 = str(x1)
            y1 = str(y1)
            x2 = str(x2)
            y2 = str(y2)
            new_image_id = image_id
        elif image_id[0] == 'h':
            x1 = float(x1)
            y1 = float(y1)
            x2 = float(x2)
            y2 = float(y2)
            if image_id[8:] == '00':
                x1 = str(x1)
                y1 = str(y1)
                x2 = str(x2)
                y2 = str(y2)
            elif image_id[8:] == '01':
                x1 = str(x1)
                y1 = str(y1+600)
                x2 = str(x2)
                y2 = str(y2+600)
            elif image_id[8:] == '02':
                x1 = str(x1)
                y1 = str(y1+1200)
                x2 = str(x2)
                y2 = str(y2+1200)
            elif image_id[8:] == '03':
                x1 = str(x1)
                y1 = str(y1+1800)
                x2 = str(x2)
                y2 = str(y2+1800)
            elif image_id[8:] == '04':
                x1 = str(x1)
                y1 = str(y1+2400)
                x2 = str(x2)
                y2 = str(y2+2400)
            elif image_id[8:] == '05':
                x1 = str(x1+800)
                y1 = str(y1)
                x2 = str(x2+800)
                y2 = str(y2)
            elif image_id[8:] == '06':
                x1 = str(x1+800)
                y1 = str(y1+600)
                x2 = str(x2+800)
                y2 = str(y2+600)
            elif image_id[8:] == '07':
                x1 = str(x1+800)
                y1 = str(y1+1200)
                x2 = str(x2+800)
                y2 = str(y2+1200)
            elif image_id[8:] == '08':
                x1 = str(x1+800)
                y1 = str(y1+1800)
                x2 = str(x2+800)
                y2 = str(y2+1800)
            elif image_id[8:] == '09':
                x1 = str(x1+800)
                y1 = str(y1+2400)
                x2 = str(x2+800)
                y2 = str(y2+2400)
            elif image_id[8:] == '10':
                x1 = str(x1+1600)
                y1 = str(y1)
                x2 = str(x2+1600)
                y2 = str(y2)
            elif image_id[8:] == '11':
                x1 = str(x1+1600)
                y1 = str(y1+600)
                x2 = str(x2+1600)
                y2 = str(y2+600)
            elif image_id[8:] == '12':
                x1 = str(x1+1600)
                y1 = str(y1+1200)
                x2 = str(x2+1600)
                y2 = str(y2+1200)
            elif image_id[8:] == '13':
                x1 = str(x1+1600)
                y1 = str(y1+1800)
                x2 = str(x2+1600)
                y2 = str(y2+1800)
            elif image_id[8:] == '14':
                x1 = str(x1+1600)
                y1 = str(y1+2400)
                x2 = str(x2+1600)
                y2 = str(y2+2400)
            elif image_id[8:] == '15':
                x1 = str(x1+2400)
                y1 = str(y1)
                x2 = str(x2+2400)
                y2 = str(y2)
            elif image_id[8:] == '16':
                x1 = str(x1+2400)
                y1 = str(y1+600)
                x2 = str(x2+2400)
                y2 = str(y2+600)
            elif image_id[8:] == '17':
                x1 = str(x1+2400)
                y1 = str(y1+1200)
                x2 = str(x2+2400)
                y2 = str(y2+1200)
            elif image_id[8:] == '18':
                x1 = str(x1+2400)
                y1 = str(y1+1800)
                x2 = str(x2+2400)
                y2 = str(y2+1800)
            elif image_id[8:] == '19':
                x1 = str(x1+2400)
                y1 = str(y1+2400)
                x2 = str(x2+2400)
                y2 = str(y2+2400)
            elif image_id[8:] == '20':
                x1 = str(x1+3200)
                y1 = str(y1)
                x2 = str(x2+3200)
                y2 = str(y2)
            elif image_id[8:] == '21':
                x1 = str(x1+3200)
                y1 = str(y1+600)
                x2 = str(x2+3200)
                y2 = str(y2+600)
            elif image_id[8:] == '22':
                x1 = str(x1+3200)
                y1 = str(y1+1200)
                x2 = str(x2+3200)
                y2 = str(y2+1200)
            elif image_id[8:] == '23':
                x1 = str(x1+3200)
                y1 = str(y1+1800)
                x2 = str(x2+3200)
                y2 = str(y2+1800)
            elif image_id[8:] == '24':
                x1 = str(x1+3200)
                y1 = str(y1+2400)
                x2 = str(x2+3200)
                y2 = str(y2+2400)
            new_image_id = image_id[1:7]
        with open('C:/Users/pcl/Desktop/Object-Detection-Metrics-master/keras_category_results_final4000nms/'+label+'.txt','a+') as ff:
            ff.write(new_image_id+" "+str(confidence)+" "+x1+" "+y1+" "+x2+" "+y2+"\n")


