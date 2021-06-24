"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose
from keras.layers.core import Dense
# import cntk as C

import keras.layers as KL


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks,  samindex, CAM = False, SAM = False):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    #sam_index = index
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        if SAM:
            maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(y)
            avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(y)
            max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
            name = "sam_conv"+str(samindex[0])
            #print(name)
            spatial_attention_feature = Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False, name=name)(max_avg_pool_spatial)
            samindex[0]+=1
            y = KL.Multiply()([y, spatial_attention_feature])

        x = Add()([x,y])
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    samindex = [0]
    x = resblock_body(x, 64, 1, samindex)
    x = resblock_body(x, 128, 2, samindex)
    x = resblock_body(x, 256, 8, samindex)
    x = resblock_body(x, 512, 8, samindex)
    x = resblock_body(x, 1024, 4, samindex)
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))
    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))
    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))
    return Model(inputs, [y1,y2,y3])
    


def yolo_body_with_classification(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))

    ''''''''''''''''''''''''''''''''''''''''''
    # add classification superivision
    classification_results = GlobalAveragePooling2D(dim_ordering='channels_last')(darknet.output)
    classification_results = Dense(1, activation='sigmoid')(classification_results)
    ''''''''''''''''''''''''''''''''''''''''''
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    # return Model(inputs, [y1,y2,y3])
    return Model(inputs, [classification_results,y1,y2,y3])



def yolo_backbone(inputs, num_anchors, num_classes):
    darknet = Model(inputs, darknet_body(inputs))
    ''''''''''''''''''''''''''''''''''''''''''
    # add classification superivision
    classification_results = GlobalAveragePooling2D(dim_ordering='channels_last')(darknet.output)
    classification_results = Dense(1, activation='sigmoid')(classification_results)
    ''''''''''''''''''''''''''''''''''''''''''
    return Model(inputs, classification_results)


def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    # box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5,
              scale=0):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    if not scale:
        for l in range(num_layers):
            _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
    # detect in one layer
    if scale:
        num_layers = 1
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[scale-1],
                anchors[anchor_mask[scale-1]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes, add_positive=False, add_positive_thresh=0.3, dual_IOU=False, SSMA=False, upper_threshold=0.7, lower_threshold=0.3):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0
    add_num = 0
    positive_num = 0
    cover_num = 0
    add_cover_num = 0
    iou_assign_list = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l])),dtype='float32') for l in range(num_layers)]

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)       
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    if y_true[l][b, j, i, k, 4] == 1:
                        cover_num += 1
                    # best anchor iou set to 1 for unchanged
                    iou_assign_list[l][b, j, i, k] = 1
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1
                    positive_num += 1

        

        if add_positive:
            layer_iou = iou.reshape((-1,3,3))
            best_layer_anchor = np.argmax(layer_iou, axis=-1)
            for t, n in enumerate(best_layer_anchor):
                for l in range(num_layers):
                    if max(layer_iou[t][2-l]) > add_positive_thresh:
                        i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                        k = best_layer_anchor[t][2-l]
                        c = true_boxes[b,t, 4].astype('int32')     
                        if y_true[l][b, j, i, k, 4] != 1:
                            add_num += 1
                            y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                            y_true[l][b, j, i, k, 4] = 1
                            y_true[l][b, j, i, k, 5+c] = 1
                            iou_assign_list[l][b, j, i, k] = max(layer_iou[t][2-l])
                        else:
                            if max(layer_iou[t][2-l]) > iou_assign_list[l][b, j, i, k]:
                                y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                                y_true[l][b, j, i, k, 4] = 1
                                y_true[l][b, j, i, k, 5:] = 0
                                y_true[l][b, j, i, k, 5+c] = 1
                                iou_assign_list[l][b, j, i, k] = max(layer_iou[t][2-l])
                                add_cover_num += 1
        if dual_IOU:
            #best_anchor = np.argmax(iou, axis=-1)
            for t, ioulist in enumerate(iou):
                for n, t_n_iou in enumerate(ioulist):
                    if t_n_iou < upper_threshold:
                        continue
                    for l in range(num_layers):
                        if n in anchor_mask[l]:
                            i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                            j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                            k = anchor_mask[l].index(n)
                            c = true_boxes[b,t, 4].astype('int32')
                            if y_true[l][b, j, i, k, 4] == 1 and t_n_iou <= iou_assign_list[l][b, j, i, k]:
                                cover_num += 1
                                break
                            
                            add_cover_num += 1
                            iou_assign_list[l][b, j, i, k] = t_n_iou
                            y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                            y_true[l][b, j, i, k, 4] = 1
                            y_true[l][b, j, i, k, 5+c] = 1
                            positive_num += 1
                            
    
    print("\npositive: {}, add: {}, cover: {} ,add_cover: {}\n".format(positive_num, add_num, cover_num, add_cover_num))
    return y_true



def preprocess_true_boxes_with_classification(true_boxes, input_shape, anchors, num_classes, add_positive=False, add_positive_thresh=0.3):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]
    y_true_classification = np.zeros((m,1), dtype='float32')

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    add_num = 0
    positive_num = 0
    cover_num = 0
    add_cover_num = 0
    iou_assign_list = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l])),dtype='float32') for l in range(num_layers)]


    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    if y_true[l][b, j, i, k, 4] == 1:
                        cover_num += 1
                    # best anchor iou set to 1 for unchanged
                    iou_assign_list[l][b, j, i, k] = 1
                    if c!=0:
                        y_true_classification[b] = 1
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1
                    positive_num += 1

                    # 对于是否可以将normal类objectness设置为极小，仍然存在疑问
                    '''
                    if c==0:
                        y_true[l][b, j, i, k, 4] = 0.001
                    '''
        if add_positive:
            layer_iou = iou.reshape((-1,3,3))
            best_layer_anchor = np.argmax(layer_iou, axis=-1)
            for t, n in enumerate(best_layer_anchor):
                for l in range(num_layers):
                    if max(layer_iou[t][2-l]) > add_positive_thresh:
                        i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                        k = best_layer_anchor[t][2-l]
                        c = true_boxes[b,t, 4].astype('int32')     
                        if y_true[l][b, j, i, k, 4] != 1:
                            add_num += 1
                            y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                            y_true[l][b, j, i, k, 4] = 1
                            y_true[l][b, j, i, k, 5+c] = 1
                            iou_assign_list[l][b, j, i, k] = max(layer_iou[t][2-l])
                        else:
                            if max(layer_iou[t][2-l]) > iou_assign_list[l][b, j, i, k]:
                                y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                                y_true[l][b, j, i, k, 4] = 1
                                y_true[l][b, j, i, k, 5:] = 0
                                y_true[l][b, j, i, k, 5+c] = 1
                                iou_assign_list[l][b, j, i, k] = max(layer_iou[t][2-l])
                                add_cover_num += 1

    print("\npositive: {}, add: {}, cover: {} ,add_cover: {}\n".format(positive_num, add_num, cover_num, add_cover_num))
    return y_true, y_true_classification


def box_iou(b1, b2, GIOU = False, DIOU = False, CIOU = False):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union = b1_area + b2_area - intersect_area
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    # 计算闭包
    union_mins = K.minimum(b1_mins, b2_mins)
    union_maxes = K.maximum(b1_maxes, b2_maxes)
    union_wh = K.maximum(union_maxes - union_mins, 0.)
    ac_area = union_wh[...,0] * union_wh[...,1]
    g_iou = iou - (ac_area - union) / ac_area

    #计算DIOU
    p_xy = b1_xy - b2_xy
    p2 = p_xy[..., 0] * p_xy[..., 0] + p_xy[..., 1] * p_xy[..., 1]
    c2 = union_wh[...,0] * union_wh[...,0] + union_wh[...,1] * union_wh[...,1] + 1e-7
    d_iou = iou - p2 / c2

    #计算CIOU
    v = (4.0 / (np.pi)**2) * tf.square((
            tf.atan((b1_wh[...,0]/b1_wh[...,1])) -
            tf.atan((b2_wh[..., 0] / b2_wh[..., 1]))))

    alpha = tf.maximum(v / (1-iou+v),0)

    c_iou= d_iou - alpha * v

    if GIOU:
        return g_iou
    if DIOU:
        return d_iou
    if CIOU:
        return c_iou 


    return iou


def yolo_loss(args, normalized_anchors, num_classes, ignore_thresh=.5, print_loss=True, focal_loss=False, add_centerness=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(normalized_anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))


    # anchors = normalized_anchors*float(K.int_shape(yolo_outputs[0])[1]*32)
    anchors = normalized_anchors
    dynamic_positive = True

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        # raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        # use dynamic positive samples instead of preset 
        dynamic_object_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)

        def loop_body(b, ignore_mask, dynamic_object_mask, dynamic_positive_threshold=0.7):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            dynamic_object_mask = dynamic_object_mask.write(b, K.cast(best_iou<dynamic_positive_threshold, K.dtype(true_box)))

            return b+1, ignore_mask, dynamic_object_mask
        

        

        _, ignore_mask, dynamic_object_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask, dynamic_object_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        if dynamic_positive:
            dynamic_object_mask = dynamic_object_mask.stack()
            dynamic_object_mask = K.expand_dims(dynamic_object_mask, -1)
            object_mask = dynamic_object_mask



        # add centerness to deal with anchor misalignment
        centerness = (1 - 0.6 * K.abs(raw_true_xy[...,0:1] - 0.5)) * (1 - 0.6 * K.abs(raw_true_xy[...,1:] - 0.5))
        # xy_loss_bonus = 2
        # confidence_loss_bonus = 2
        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        if not focal_loss:
            confidence_loss = object_mask * centerness * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
                (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        else:
            confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
                (1-object_mask) * K.square(raw_pred[...,4:5]) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)
        
        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf

        reg_loss=object_mask*(1-box_iou(pred_box, y_true[l][..., :4])) 
        reg_loss = K.sum(reg_loss) / mf

        # loss += xy_loss + wh_loss + confidence_loss + class_loss
        loss += reg_loss + confidence_loss + class_loss
        # loss += sum_on_image(xy_loss) + sum_on_image(wh_loss) + sum_on_image(confidence_loss) + sum_on_image(class_loss)
        
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
        #loss.shape=()
    return loss



def sum_on_image(loss):
    image_loss = loss
    for i in range(4):
        image_loss = K.sum(image_loss, axis=-1)
    return image_loss


def yolo_loss_with_classification(args, normalized_anchors, num_classes, ignore_thresh=.5, print_loss=True):
    num_layers = len(normalized_anchors)//3
    yolo_outputs_classification = args[0]
    yolo_outputs = args[1:num_layers+1]
    y_true = args[1+num_layers:2*num_layers+1]
    y_true_classification = args[-1]
    detection_mask = y_true_classification
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for i in range(3):
        detection_mask = K.expand_dims(detection_mask, -1)

    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss_detection = 0


    # anchors = normalized_anchors*float(K.int_shape(yolo_outputs[0])[1]*32)
    anchors = normalized_anchors
 
    for l in range(num_layers):

        object_mask = y_true[l][..., 4:5]      
        
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        # raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)



        # xy_loss_bonus = 2
        # confidence_loss_bonus = 2
        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            detection_mask * (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        
        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss_detection += xy_loss + wh_loss + confidence_loss + class_loss
        
        # loss += sum_on_image(xy_loss) + sum_on_image(wh_loss) + sum_on_image(confidence_loss) + sum_on_image(class_loss)
        '''
        if print_loss:
            loss_detection = tf.Print(loss_detection, [loss_detection, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss_detection: ')
        #loss.shape=()
        '''
    

    #outputc = C.clip(yolo_outputs_classification, 1e-7, 1.0 - 1e-7)
    #outputc = -y_true_classification * C.log(outputc) - 2 * (1.0 - y_true_classification) * C.log(1.0 - outputc)
    loss_classification = K.sum(K.binary_crossentropy(y_true_classification, yolo_outputs_classification, from_logits=False)) / mf
    #loss_classification = K.sum(outputc) / mf
    loss = loss_detection + loss_classification #shape=()
    if print_loss:
        loss = tf.Print(loss, [loss, loss_detection, loss_classification, detection_mask], message='loss: ')
    return loss




def assign_detection_mask(detection_mask, y_true_classification):
    m = K.shape(detection_mask)[0]
    for b in range(m):
        if y_true_classification[b]==1:
            detection_mask = detection_mask[b].assign(K.ones_like(detection_mask[0]))
    return detection_mask

