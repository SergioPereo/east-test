import os

import time
import datetime
import cv2
import numpy as np
import uuid
import json
import pytesseract

import functools
import logging
import collections

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@functools.lru_cache(maxsize=100)
def get_predictor(checkpoint_path):
    logger.info('loading model')
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import model
    from icdar import restore_rectangle
    import lanms
    from eval import resize_image, sort_poly, detect

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
    logger.info('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)

    def predictor(img):
        """
        :return: {
            'text_lines': [
                {
                    'score': ,
                    'x0': ,
                    'y0': ,
                    'x1': ,
                    ...
                    'y3': ,
                }
            ],
            'rtparams': {  # runtime parameters
                'image_size': ,
                'working_size': ,
            },
            'timing': {
                'net': ,
                'restore': ,
                'nms': ,
                'cpuinfo': ,
                'meminfo': ,
                'uptime': ,
            }
        }
        """
        start_time = time.time()
        rtparams = collections.OrderedDict()
        rtparams['start_time'] = datetime.datetime.now().isoformat()
        rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])

        im_resized, (ratio_h, ratio_w) = resize_image(img)
        rtparams['working_size'] = '{}x{}'.format(
            im_resized.shape[1], im_resized.shape[0])
        start = time.time()
        score, geometry = sess.run(
            [f_score, f_geometry],
            feed_dict={input_images: [im_resized[:,:,::-1]]})
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
        logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

        if boxes is not None:
            scores = boxes[:,8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        timer['overall'] = duration
        logger.info('[timing] {}'.format(duration))

        text_lines = []
        if boxes is not None:
            text_lines = []
            for box, score in zip(boxes, scores):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue
                tl = collections.OrderedDict(zip(
                    ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                    map(float, box.flatten())))
                tl['score'] = float(score)
                text_lines.append(tl)
        ret = {
            'text_lines': text_lines,
            'rtparams': rtparams,
            'timing': timer,
        }
        #ret.update(get_host_info())
        return ret


    return predictor

def trunc(number):
    if number < 0:
        return 0
    else:
        return number

def save_result(img, rst, image_path, filename):
    rows = []
    for index, t in enumerate(rst['text_lines']):

        bounding = np.array([[trunc(t['x0']), trunc(t['y0'])], [trunc(t['x1']), trunc(t['y1'])], [trunc(t['x2']),
                      trunc(t['y2'])], [trunc(t['x3']), trunc(t['y3'])]], dtype='int32')
        bounding = bounding.reshape(-1, 2)
        d = np.array([[(trunc(t['x0']),trunc(t['y0'])), (trunc(t['x1']),trunc(t['y1'])), (trunc(t['x2']),trunc(t['y2'])), (trunc(t['x3']), trunc(t['y3']))]], dtype=np.int32)
        rect = cv2.boundingRect(bounding)
        x,y,w,h = rect

        height = img.shape[0]
        width = img.shape[1]

        mask = np.zeros((height, width), dtype=np.uint8)

        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(mask, d, ignore_mask_color)

        res = cv2.bitwise_and(img,img,mask = mask)

        cropped = res[y:y+h, x:x+w].copy()
        image_string = pytesseract.image_to_string(cropped)
        image_string = image_string.replace('\n', '')
        rows.append(f'{str(index) + "_" + filename},{image_string}\n')
        cv2.imwrite(os.path.join(image_path, (str(index) + "_" + filename)), cropped)

    return rows

def predict(image_path, filename, create_new_file, images_path):
    global predictor
    img = cv2.imread(image_path)
    rst = get_predictor(checkpoint_path)(img)
    rows = save_result(img, rst, images_path, filename)
    if create_new_file:
        with open((filename.split('.')[0] + ".csv"), 'w') as f:
            for row in rows:
                f.write(row)
    else:
        with open((filename.split('.')[0] + ".csv"), 'a') as f:
            for row in rows:
                f.write(row)

def main():
    global checkpoint_path
    current_dir = os.getcwd()
    checkpoint_path = os.path.join(os.path.join(current_dir,"pretrain"), "east_icdar2015_resnet_v1_50_rbox")
    images_path = os.path.join(current_dir, 'images')

    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            'Checkpoint `{}` not found'.format(checkpoint_path))

    for index, file in enumerate(os.listdir(images_path)):
        filename = os.fsdecode(file)
        image_path = os.path.join(images_path, filename)
        predict(image_path, filename, index==0, images_path)

if __name__ == '__main__':
    main()