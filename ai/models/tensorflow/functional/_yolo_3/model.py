"""
    Yolo v3 implementation pre-trained on COCO.

    2019 Colin Torney
    
    Based on code from https://github.com/experiencor/keras-yolo3
    MIT License Copyright (c) 2017 Ngoc Anh Huynh

    Modified to process output within tensorflow and be agnostic to image size

"""


from datetime import datetime

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Dense, Flatten, Activation, Reshape, Lambda
from tensorflow.keras.layers import add, concatenate
import tensorflow as tf

from tensorflow.keras import backend as K

ANC_VALS = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]

def _conv_block(inp, convs, skip=True, train=False):
    x = inp
    count = 0
    
    for conv in convs:
        if count == (len(convs) - 2) and skip:
            skip_connection = x
        count += 1
        
        if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
        if 'train' in conv:
            trainflag=conv['train']#update the value for the key
        else:
            trainflag=train
        x = Conv2D(conv['filter'], 
                   conv['kernel'], 
                   strides=conv['stride'], 
                   padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
                   name='conv_' + str(conv['layer_idx']), 
                   use_bias=False if conv['bnorm'] else True, trainable=trainflag)(x)
        if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']),trainable=trainflag)(x)
        if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']),trainable=trainflag)(x)

    return add([skip_connection, x]) if skip else x

def crop(start, end):
    # Crops (or slices) a Tensor on fourth dimension from start to end
    def func(x):
        return x[:, :, :, :, start: end]
    return Lambda(func)

def anchors(i):
    def func(x):
        anc = tf.constant(ANC_VALS[i], dtype='float', shape=[1,1,1,3,2])
        return tf.exp(x) * anc 
    return Lambda(func)

def positions():
    def func(z):
        x = z[0]
        y = z[1]
        # compute grid factor and net factor
        grid_h      = tf.shape(x)[1]
        grid_w      = tf.shape(x)[2]

        im_h      = tf.shape(y)[1]
        im_w      = tf.shape(y)[2]

        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])
        net_factor  = tf.reshape(tf.cast([im_w, im_h], tf.float32), [1,1,1,1,2])
        
        cell_x = tf.cast(tf.reshape(tf.tile(tf.range(tf.maximum(grid_h,grid_w)), [tf.maximum(grid_h,grid_w)]), (1, tf.maximum(grid_h,grid_w), tf.maximum(grid_h,grid_w), 1, 1)),dtype=tf.float32)

        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [1, 1, 1, 3, 1])
        pred_box_xy = (cell_grid[:,:grid_h,:grid_w,:,:] + x) 
        pred_box_xy = pred_box_xy * net_factor/grid_factor 

        return pred_box_xy 
    return Lambda(func)

def reshape_last_layer(out_size):
    def func(x):
        # reshape last 2 dimensions 
        in_b      = tf.shape(x)[0]
        in_h      = tf.shape(x)[1]
        in_w      = tf.shape(x)[2]
        
        final_l = tf.reshape(x, [in_b, in_h, in_w, 3, out_size])
        return final_l

    return Lambda(func)


def get_yolo_model(num_class=80, trainable=False, headtrainable=False):

    # for each box we have num_class outputs, 4 bbox coordinates, and 1 object confidence value
    out_size = num_class+5
    input_image = Input(shape=(None, None, 3))

    # Layer  0 => 4
    x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}], train=trainable)

    # Layer  5 => 8
    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}], train=trainable)

    # Layer  9 => 11
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}], train=trainable)

    # Layer 12 => 15
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}], train=trainable)

    # Layer 16 => 36
    for i in range(7):
        x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}], train=trainable)
        
    skip_36 = x
        
    # Layer 37 => 40
    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}], train=trainable)

    # Layer 41 => 61
    for i in range(7):
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}], train=trainable)
        
    skip_61 = x
        
    # Layer 62 => 65
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}], train=trainable)

    # Layer 66 => 74
    for i in range(3):
        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}], train=trainable)
        
    # Layer 75 => 79
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False, train=trainable)

    # Layer 80 => 82
    if num_class!=80:
        yolo_82 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                              {'filter':  3*out_size, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,'train': headtrainable, 'layer_idx': 981}], skip=False, train=trainable)
    else:
        yolo_82 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                              {'filter':  3*out_size, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,'train': headtrainable, 'layer_idx': 81}], skip=False, train=trainable)

    # Layer 83 => 86
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False, train=trainable)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])

    # Layer 87 => 91
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False, train=trainable)

    # Layer 92 => 94
    if num_class!=80:
        yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
                    {'filter': 3*out_size, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'train': headtrainable, 'layer_idx': 993}], skip=False, train=trainable)
    else:
        yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
                    {'filter': 3*out_size, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'train': headtrainable, 'layer_idx': 93}], skip=False, train=trainable)

    # Layer 95 => 98
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False, train=trainable)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])

    # Layer 99 => 106
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},], skip=False, train=trainable)

    if num_class!=80:
        yolo_106 = _conv_block(x, [{'filter': 3*out_size, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'train': headtrainable,'layer_idx': 9105}], skip=False, train=trainable)
    else:
        yolo_106 = _conv_block(x, [{'filter': 3*out_size, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'train': headtrainable,'layer_idx': 105}], skip=False, train=trainable)


    # TEST CODE TIDY UP #

    final_layers = [yolo_82, yolo_94, yolo_106]
    output = []
    l_anchor = 0

    for fl in final_layers:

        final_shaped = reshape_last_layer(out_size)(fl)
        # process centre points for grid offsets and convert to image coordinates
        l_offs = crop(0,2)(final_shaped)
        l_offs = Activation('sigmoid')(l_offs)
        l_offs = positions()([l_offs, input_image])

        # process anchor boxes
        l_szs = crop(2,4)(final_shaped)
        l_szs = anchors(l_anchor)(l_szs)
        l_anchor+=1

        # object confidence
        l_obj = crop(4,5)(final_shaped)
        l_obj = Activation('sigmoid')(l_obj)

        # class scores
        l_cls = crop(5,out_size)(final_shaped)
        l_cls = Activation('softmax')(l_cls)

        # combine results
        l_out = concatenate([l_offs, l_szs, l_obj, l_cls])
        output.append(l_out)

    model = Model(input_image,output)
    return model



class yolo_model():

    def __init__(self, labelclassMap, state, pretrained=True, alltrain=False):

        self.labelclassMap = labelclassMap
        self.numClasses = len(labelclassMap.keys())
        self.pretrained = pretrained
        self.alltrain = alltrain
        self.yolo_nn = get_yolo_model(self.numClasses, trainable=self.alltrain, headtrainable=True)

        timestampStr = datetime.now().strftime("%Y%m%d%H%M%S")

        self.state = (state if state is not None else 'weights/' + timestampStr)




        if self.pretrained: 
            print('loading pretrained weights')
            self.yolo_nn.load_weights('weights/yolo-v3-coco.h5', by_name=True) #<- Fix this to get weights from somewhere more accessible<-!



    def getStateDict(self):
        if self.state is not None:
            self.yolo_nn.save_weights(self.state + '.h5')
            self.yolo_nn.save(self.state + '_full.h5')

        stateDict = {
            'model_state': self.state, 
            'labelclassMap': self.labelclassMap,
            'pretrained': self.pretrained,
            'alltrain': self.alltrain
        }
        return stateDict


    @staticmethod
    def loadFromStateDict(stateDict):
        # parse args
        labelclassMap = stateDict['labelclassMap']
        pretrained = (stateDict['pretrained'] if 'pretrained' in stateDict else True)
        alltrain = (stateDict['alltrain'] if 'alltrain' in stateDict else False)
        state = (stateDict['model_state'] if 'model_state' in stateDict else None)
        init_weights = (stateDict['init_weights'] if 'init_weights' in stateDict else None)

        # return model
        model = yolo_model(labelclassMap, state, pretrained if state is None else False)
        if state is not None:
            print('loading saved weights')
            model.yolo_nn.load_weights(state + '.h5')
        elif init_weights is not None:
            print('loading initial weights')
            model.yolo_nn.load_weights(init_weights)
        return model


