import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import xml.etree.ElementTree as ET
from os import getcwd

sets=[('ury', 'train'), ('ury', 'val'), ('ury', 'test')]

#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#classes = ["normal","ascus","asch","lsil","hsil","agc","adenocarcinoma","vaginalis","monilia","dysbacteriosis"]
classes=["eryth", "leuko", "epith", "cryst", "cast", "mycete", "epithn"]

def convert_annotation(image_set, image_id, list_file):
    in_file = open('/home/data/ury_data/annotations/%s/%s.xml'%(image_set, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

# wd = getcwd()

image_set = 'train'
year = 'ury'
for year, image_set in sets:
    image_ids = open('/home/data/ury_data/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('/home/data/ury_data/%s/%s.jpg'%(image_set,image_id))
        convert_annotation(image_set, image_id, list_file)
        list_file.write('\n')
    list_file.close()

