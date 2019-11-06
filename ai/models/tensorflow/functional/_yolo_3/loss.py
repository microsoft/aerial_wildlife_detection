import tensorflow as tf



class YoloLoss():
    def __init__(self, width, height, NO_OBJECT_SCALE=1.0, OBJECT_SCALE=1.0, COORD_SCALE=1.0, CLASS_SCALE=1.0):
        self.IMAGE_W = width
        self.IMAGE_H = height
        self.NO_OBJECT_SCALE = NO_OBJECT_SCALE
        self.OBJECT_SCALE = OBJECT_SCALE
        self.COORD_SCALE = COORD_SCALE
        self.CLASS_SCALE = CLASS_SCALE

    def yolo_loss(self, y_true, y_pred):
        # compute grid factor and net factor
        grid_h = tf.shape(y_true)[1]
        grid_w = tf.shape(y_true)[2]

        grid_factor = tf.reshape(
            tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

        net_h = self.IMAGE_H / grid_h
        net_w = self.IMAGE_W / grid_w
        net_factor = tf.reshape(
            tf.cast([net_w, net_h], tf.float32), [1, 1, 1, 1, 2])

        pred_box_xy = y_pred[..., 0:2]  # t_wh
        pred_box_wh = tf.math.log(y_pred[..., 2:4])  # t_wh
        pred_box_conf = tf.expand_dims(y_pred[..., 4], 4)
        pred_box_class = y_pred[..., 5:]  # adjust class probabilities
        # initialize the masks
        object_mask = tf.expand_dims(y_true[..., 4], 4)

        true_box_xy = y_true[..., 0:2]  # (sigma(t_xy) + c_xy)
        true_box_wh = tf.where(y_true[..., 2:4] > 0,
                               tf.math.log(tf.cast(y_true[..., 2:4], tf.float32)),
                               y_true[..., 2:4])
        true_box_conf = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = y_true[..., 5:]

        xy_delta = self.COORD_SCALE * object_mask * (pred_box_xy - true_box_xy)  #/net_factor #* xywh_scale
        wh_delta = self.COORD_SCALE * object_mask * (pred_box_wh - true_box_wh)  #/ net_factor #* xywh_scale

        obj_delta = self.OBJECT_SCALE * object_mask * (
            pred_box_conf - true_box_conf)
        no_obj_delta = self.NO_OBJECT_SCALE * (1 - object_mask) * pred_box_conf
        class_delta = self.CLASS_SCALE * object_mask * (
            pred_box_class - true_box_class)

        loss_xy = tf.reduce_sum(tf.square(xy_delta), list(range(1, 5)))
        loss_wh = tf.reduce_sum(tf.square(wh_delta), list(range(1, 5)))
        loss_obj = tf.reduce_sum(tf.square(obj_delta), list(range(1, 5)))
        lossnobj = tf.reduce_sum(tf.square(no_obj_delta), list(range(1, 5)))
        loss_cls = tf.reduce_sum(tf.square(class_delta), list(range(1, 5)))

        loss = loss_xy + loss_wh + loss_obj + lossnobj + loss_cls
        return loss

