
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json

''' 
class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
'''
class_names = ["normal","ascus","asch","lsil","hsil","agc","adenocarcinoma","vaginalis","monilia","dysbacteriosis"]


def voc2coco(data_dir, train_file, test_file):
    xml_dir = os.path.join(data_dir, 'Annotations')
    img_dir = os.path.join(data_dir, 'JPEGImages')
 
    with open(train_file, 'r') as f:
        train_fs = f.readlines()
    train_xmls = [os.path.join(xml_dir, n.strip() + '.xml') for n in train_fs]
    with open(test_file, 'r') as f:
        test_fs = f.readlines()
    test_xmls = [os.path.join(xml_dir, n.strip() + '.xml') for n in test_fs]
    print('got xmls')
    train_coco = xml2coco(train_xmls)
    test_coco = xml2coco(test_xmls)
    with open(os.path.join(data_dir, 'coco_train.json'), 'w') as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=2)
    with open(os.path.join(data_dir, 'coco_test.json'), 'w') as f:
        json.dump(test_coco, f, ensure_ascii=False, indent=2)
    print('done')
 
 
 
def xml2coco(xmls):
    coco_anno = {'info': {}, 'images': [], 'licenses': [], 'annotations': [], 'categories': []}
    coco_anno['categories'] = [{'supercategory': j, 'id': i+1, 'name': j} for i,j in enumerate(class_names)]
    img_id = 0
    anno_id = 0
    for fxml in tqdm(xmls):
        try:
            tree = ET.parse(fxml)
            objects = tree.findall('object')
        except:
            print('err xml file: ', fxml)
            continue
        if len(objects) < 1:
            print('no object in ', fxml)
            continue
        img_id += 1
        size = tree.find('size')
        ih = float(size.find('height').text)
        iw = float(size.find('width').text)
        img_name = fxml.strip().split('/')[-1].replace('xml', 'jpg')
        img_info = {}
        img_info['id'] = img_id
        img_info['file_name'] = img_name
        img_info['height'] = ih
        img_info['width'] = iw
        coco_anno['images'].append(img_info)
 
        for obj in objects:
            cls_name = obj.find('name').text
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            if x2 < x1 or y2 < y1:
                print('bbox not valid: ', fxml)
                continue
            anno_id += 1
            bb = [x1, y1, x2 - x1, y2 - y1]
            categery_id = class_names.index(cls_name) + 1
            area = (x2 - x1) * (y2 - y1)
            anno_info = {}
            anno_info['segmentation'] = []
            anno_info['area'] = area
            anno_info['image_id'] = img_id
            anno_info['bbox'] = bb
            anno_info['iscrowd'] = 0
            anno_info['category_id'] = categery_id
            anno_info['id'] = anno_id
            coco_anno['annotations'].append(anno_info)
 
    return coco_anno
 
 
if __name__ == '__main__':
    data_dir = '/home/gp/work/project/learning/VOC/VOCdevkit/VOC2007'
    train_file = '/home/gp/work/project/learning/VOC/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
    test_file = '/home/gp/work/project/learning/VOC/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    voc2coco(data_dir, train_file, test_file)
