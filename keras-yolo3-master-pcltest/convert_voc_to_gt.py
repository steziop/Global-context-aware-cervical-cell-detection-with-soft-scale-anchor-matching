import xml.etree.ElementTree as ET
from os import getcwd

def convert_voc_to_gt(classnames, voc_dir, gt_dir):
    with open(voc_dir) as f:
        imglines = f.readlines()
    for imgline in imglines:
        imgpath = imgline.split()[0]
        image_id = imgpath.split('/')[-1].split('.')[0]
        gtboxes = imgline.split()[1:]
        image_gt_file = open(gt_dir+'/%s.txt'%(image_id),'w')
        for gtbox in gtboxes:
            left, top, right, bottom, class_id = gtbox.split(',')
            left = round(float(left))
            top = round(float(top))
            right = round(float(right))
            bottom = round(float(bottom))
            class_id = int(class_id)
            label = classnames[class_id]
            image_gt_file.write(label+" "+str(left)+" "+str(top)+" "+str(right)+" "+str(bottom)+"\n")
        image_gt_file.close()



if __name__ == '__main__':
    wd = getcwd()
    with open(wd+'/model_data/my_classes.txt') as fclass:
        classnames = fclass.read().rstrip().split()
    
    voc_dir = input('Input test file path:')
    gt_dir = input('Input Ground Truth directory:')

    convert_voc_to_gt(classnames, voc_dir, gt_dir)