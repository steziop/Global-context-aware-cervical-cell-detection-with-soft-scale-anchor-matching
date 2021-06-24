import numpy as np
import os

for i in range(100):
    rootdir = '/home/data/TCT_data100/testdata/00' + str(1500000+i+1) + '/TCT/detections_4k/'
    nmsdir = '/home/data/TCT_data100/testdata/00' + str(1500000+i+1) + '/TCT/detections_4k_nms/'
    if os.path.exists(nmsdir):
        continue
    else:
        os.mkdir(nmsdir)
    filenames = os.listdir(rootdir)
    for filename in filenames:
        detfile=rootdir+filename
        #print(detfile)
        dets=np.loadtxt(detfile,dtype=bytes).astype(str)            
        #print(dets)
        if dets.size == 6:
            dets = np.expand_dims(dets,0)
        thresh = 0.3
        x1 = dets[:, 2].astype(float)
        y1 = dets[:, 3].astype(float)
        x2 = dets[:, 4].astype(float)
        y2 = dets[:, 5].astype(float)
        scores = dets[:, 1].astype(float)
        #print(x1,x1.dtype)
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) # 每个boundingbox的面积
        #print(areas)
        order = scores.argsort()[::-1] # boundingbox的置信度排序
        keep = [] # 用来保存最后留下来的boundingbox
        while order.size > 0:     
            i = order[0] # 置信度最高的boundingbox的index
            keep.append(i) # 添加本次置信度最高的boundingbox的index
            
            
            # 当前bbox和剩下bbox之间的交叉区域
            # 选择大于x1,y1和小于x2,y2的区域
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #保留交集小于一定阈值的boundingbox
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
            #print(keep)
        #return keep
        #print(keep)
        for k in keep:
            #dets_new=dets[i,:].tolist()
            lines=' '.join(dets[k])
            #print(lines)
            line=lines.split('\n')
            #print(line)
            l = line[0]
            #print(l)
            image_id,scores,x1,y1,x2,y2 = l.split()
            #print(scores)
            with open(nmsdir + filename,'a+') as ff:
                ff.writelines(str(image_id)+" "+scores+" "+x1+" "+y1+" "+x2+" "+y2+"\n")



