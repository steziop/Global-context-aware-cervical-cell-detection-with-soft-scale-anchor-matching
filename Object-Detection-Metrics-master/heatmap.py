import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body, yolo_body_with_classification, yolo_backbone

import matplotlib.pyplot as plt

K.clear_session()

model = yolo_body_with_classification(Input(shape=(None,None,3)), 9//3, 10)
model.load_weights("yolo_with_classification.h5") # make sure model, anchors and classes match

#print(model.summary())



img_path = 'E:\\VOC2019\\VOC2019\\JPEGImages\\ht07612_20.jpg'

img = image.load_img(img_path, target_size=(416, 416))   # 大小为800*600的Python图像库图像

x = image.img_to_array(img)  # 形状为（416， 416， 3）的float32格式Numpy数组

x = np.expand_dims(x, axis=0)  # 添加一个维度，将数组转化为（1， 416， 416， 3）的形状批量

x = preprocess_input(x)   #按批量进行预处理（按通道颜色进行标准化）


preds = model.predict(x)

classification = model.output[0]
#print('Predicted:', decode_predictions(preds, top=3)[0])





last_conv_layer = model.get_layer('add_23')  # add_23层的输出特征图

grads = K.gradients(classification, last_conv_layer.output)[0]   # 非洲象类别相对于block5_conv3输出特征图的梯度

pooled_grads = K.mean(grads, axis=(0, 1, 2))   # 形状是（512， ）的向量，每个元素是特定特征图通道的梯度平均大小

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])  # 这个函数允许我们获取刚刚定义量的值：对于给定样本图像，pooled_grads和block5_conv3层的输出特征图

pooled_grads_value, conv_layer_output_value = iterate([x])  # 给我们两个大象样本图像，这两个量都是Numpy数组

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]  # 将特征图数组的每个通道乘以这个通道对大象类别重要程度

heatmap = np.mean(conv_layer_output_value, axis=-1)  # 得到的特征图的逐通道的平均值即为类激活的热力图





heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()