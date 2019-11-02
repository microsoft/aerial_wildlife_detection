
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

def positions(h,w):
    def func(x):
        # compute grid factor and net factor
        grid_h      = tf.shape(x)[1]
        grid_w      = tf.shape(x)[2]

        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])
        net_factor  = tf.reshape(tf.cast([w, h], tf.float32), [1,1,1,1,2])
        
        cell_x = tf.cast(tf.reshape(tf.tile(tf.range(tf.maximum(grid_h,grid_w)), [tf.maximum(grid_h,grid_w)]), (1, tf.maximum(grid_h,grid_w), tf.maximum(grid_h,grid_w), 1, 1)),dtype=tf.float32)

        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [1, 1, 1, 3, 1])
        pred_box_xy = (cell_grid[:,:grid_h,:grid_w,:,:] + x) 
        pred_box_xy = pred_box_xy * net_factor/grid_factor 

        return pred_box_xy 
    return Lambda(func)

def get_yolo_model(in_w=416,in_h=416, num_class=80, trainable=False, headtrainable=False):

    # for each box we have num_class outputs, 4 bbox coordinates, and 1 object confidence value
    out_size = num_class+5
    input_image = Input(shape=( in_h,in_w, 3))

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


    final_large = Reshape((in_h//32,in_w//32,3,out_size))(yolo_82)
    final_med = Reshape((in_h//16, in_w//16,3,out_size))(yolo_94)
    final_small = Reshape((in_h//8,in_w//8,3,out_size))(yolo_106)
    #output = [final_large, final_med, final_small]  
    #model = Model(input_image,output)
    #return model

    s_offs =crop(0,2)(final_small)
    s_szs =crop(2,4)(final_small)
    s_obj =crop(4,5)(final_small)
    s_obj = Activation('sigmoid')(s_obj)
    s_cls =crop(5,out_size)(final_small)
    s_cls = Activation('softmax')(s_cls)
    s_szs = anchors(2)(s_szs)
    s_offs = Activation('sigmoid')(s_offs)
    s_offs = positions(in_h,in_w)(s_offs)
    s_out = concatenate([s_offs, s_szs, s_obj, s_cls])

    m_offs =crop(0,2)(final_med)
    m_szs =crop(2,4)(final_med)
    m_obj =crop(4,5)(final_med)
    m_obj = Activation('sigmoid')(m_obj)
    m_cls =crop(5,out_size)(final_med)
    m_cls = Activation('softmax')(m_cls)
    m_szs = anchors(1)(m_szs)
    m_offs = Activation('sigmoid')(m_offs)
    m_offs = positions(in_h,in_w)(m_offs)
    m_out = concatenate([m_offs, m_szs, m_obj, m_cls])

    l_offs =crop(0,2)(final_large)
    l_szs =crop(2,4)(final_large)
    l_obj =crop(4,5)(final_large)
    l_obj = Activation('sigmoid')(l_obj)
    l_cls =crop(5,out_size)(final_large)
    l_cls = Activation('softmax')(l_cls)
    l_szs = anchors(0)(l_szs)
    l_offs = Activation('sigmoid')(l_offs)
    l_offs = positions(in_h,in_w)(l_offs)
    l_out = concatenate([l_offs, l_szs, l_obj, l_cls])

    output = [l_out, m_out, s_out]  

    model = Model(input_image,output)
    return model





