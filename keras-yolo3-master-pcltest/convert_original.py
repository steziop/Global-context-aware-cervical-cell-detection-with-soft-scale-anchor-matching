import os
import shutil

#rootdir = '/root/userfolder/Object-Detection-Metrics/detections/'

def convert(rootdir):
    filenames = os.listdir(rootdir + "detections/")
    for filename in filenames:
        image_id = filename.split('.')[0]
        #print(image_id)
        with open(rootdir + "detections/" + filename,'r') as f1:
            lines = f1.readlines()
        for line in lines:
            label, confidence, x1, y1, x2, y2 = line.split()
            if label =='aschsil' or label =='asclsil' or label =='gland' or label =='microbial':#过滤超类
                continue
            if image_id[0:1] == 'l':
                x1 = float(x1)*2.5
                y1 = float(y1)*2.5
                x2 = float(x2)*2.5
                y2 = float(y2)*2.5
                if image_id.split('_')[-1] == '00':
                    x1 = str(x1)
                    y1 = str(y1)
                    x2 = str(x2)
                    y2 = str(y2)
                elif image_id.split('_')[-1] == '01':
                    x1 = str(x1)
                    y1 = str(y1+1500)
                    x2 = str(x2)
                    y2 = str(y2+1500)
                elif image_id.split('_')[-1] == '02':
                    x1 = str(x1+2000)
                    y1 = str(y1)
                    x2 = str(x2+2000)
                    y2 = str(y2)
                elif image_id.split('_')[-1] == '03':
                    x1 = str(x1+2000)
                    y1 = str(y1+1500)
                    x2 = str(x2+2000)
                    y2 = str(y2+1500)
                new_image_id = image_id[1:-3]
            elif len(image_id) < 7:
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
                if image_id.split('_')[-1] == '00':
                    x1 = str(x1)
                    y1 = str(y1)
                    x2 = str(x2)
                    y2 = str(y2)
                elif image_id.split('_')[-1] == '01':
                    x1 = str(x1)
                    y1 = str(y1+600)
                    x2 = str(x2)
                    y2 = str(y2+600)
                elif image_id.split('_')[-1] == '02':
                    x1 = str(x1)
                    y1 = str(y1+1200)
                    x2 = str(x2)
                    y2 = str(y2+1200)
                elif image_id.split('_')[-1] == '03':
                    x1 = str(x1)
                    y1 = str(y1+1800)
                    x2 = str(x2)
                    y2 = str(y2+1800)
                elif image_id.split('_')[-1] == '04':
                    x1 = str(x1)
                    y1 = str(y1+2400)
                    x2 = str(x2)
                    y2 = str(y2+2400)
                elif image_id.split('_')[-1] == '05':
                    x1 = str(x1+800)
                    y1 = str(y1)
                    x2 = str(x2+800)
                    y2 = str(y2)
                elif image_id.split('_')[-1] == '06':
                    x1 = str(x1+800)
                    y1 = str(y1+600)
                    x2 = str(x2+800)
                    y2 = str(y2+600)
                elif image_id.split('_')[-1] == '07':
                    x1 = str(x1+800)
                    y1 = str(y1+1200)
                    x2 = str(x2+800)
                    y2 = str(y2+1200)
                elif image_id.split('_')[-1] == '08':
                    x1 = str(x1+800)
                    y1 = str(y1+1800)
                    x2 = str(x2+800)
                    y2 = str(y2+1800)
                elif image_id.split('_')[-1] == '09':
                    x1 = str(x1+800)
                    y1 = str(y1+2400)
                    x2 = str(x2+800)
                    y2 = str(y2+2400)
                elif image_id.split('_')[-1] == '10':
                    x1 = str(x1+1600)
                    y1 = str(y1)
                    x2 = str(x2+1600)
                    y2 = str(y2)
                elif image_id.split('_')[-1] == '11':
                    x1 = str(x1+1600)
                    y1 = str(y1+600)
                    x2 = str(x2+1600)
                    y2 = str(y2+600)
                elif image_id.split('_')[-1] == '12':
                    x1 = str(x1+1600)
                    y1 = str(y1+1200)
                    x2 = str(x2+1600)
                    y2 = str(y2+1200)
                elif image_id.split('_')[-1] == '13':
                    x1 = str(x1+1600)
                    y1 = str(y1+1800)
                    x2 = str(x2+1600)
                    y2 = str(y2+1800)
                elif image_id.split('_')[-1] == '14':
                    x1 = str(x1+1600)
                    y1 = str(y1+2400)
                    x2 = str(x2+1600)
                    y2 = str(y2+2400)
                elif image_id.split('_')[-1] == '15':
                    x1 = str(x1+2400)
                    y1 = str(y1)
                    x2 = str(x2+2400)
                    y2 = str(y2)
                elif image_id.split('_')[-1] == '16':
                    x1 = str(x1+2400)
                    y1 = str(y1+600)
                    x2 = str(x2+2400)
                    y2 = str(y2+600)
                elif image_id.split('_')[-1] == '17':
                    x1 = str(x1+2400)
                    y1 = str(y1+1200)
                    x2 = str(x2+2400)
                    y2 = str(y2+1200)
                elif image_id.split('_')[-1] == '18':
                    x1 = str(x1+2400)
                    y1 = str(y1+1800)
                    x2 = str(x2+2400)
                    y2 = str(y2+1800)
                elif image_id.split('_')[-1] == '19':
                    x1 = str(x1+2400)
                    y1 = str(y1+2400)
                    x2 = str(x2+2400)
                    y2 = str(y2+2400)
                elif image_id.split('_')[-1] == '20':
                    x1 = str(x1+3200)
                    y1 = str(y1)
                    x2 = str(x2+3200)
                    y2 = str(y2)
                elif image_id.split('_')[-1] == '21':
                    x1 = str(x1+3200)
                    y1 = str(y1+600)
                    x2 = str(x2+3200)
                    y2 = str(y2+600)
                elif image_id.split('_')[-1] == '22':
                    x1 = str(x1+3200)
                    y1 = str(y1+1200)
                    x2 = str(x2+3200)
                    y2 = str(y2+1200)
                elif image_id.split('_')[-1] == '23':
                    x1 = str(x1+3200)
                    y1 = str(y1+1800)
                    x2 = str(x2+3200)
                    y2 = str(y2+1800)
                elif image_id.split('_')[-1] == '24':
                    x1 = str(x1+3200)
                    y1 = str(y1+2400)
                    x2 = str(x2+3200)
                    y2 = str(y2+2400)
                new_image_id = image_id[1:-3]
            #with open('/root/userfolder/Object-Detection-Metrics/detections_original/'+new_image_id+'.txt','a+') as ff:
            with open(rootdir + "detections_4k/" + new_image_id + '.txt','a+') as ff:
                ff.write(label+" "+str(confidence)+" "+x1+" "+y1+" "+x2+" "+y2+"\n")



for i in range(100):
    rootdir ='/home/data/TCT_data100/testdata/00' + str(1500000 + i+1) + '/TCT/'
    d4kdir = rootdir + "detections_4k/"
    shutil.rmtree(d4kdir)

    if not os.path.exists(d4kdir):
        os.mkdir(d4kdir)
    convert(rootdir)