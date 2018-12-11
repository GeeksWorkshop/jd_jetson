#! /usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import os
import numpy as np

CUR_DIR = os.path.abspath('./')
print(CUR_DIR)
SAVE_DIR = 'src/demo/scripts/net/pb'
SAVE_PATH = os.path.join(CUR_DIR, SAVE_DIR, 'flowers_mobilenet.pb')
print(SAVE_PATH)

sess = tf.Session()
output_graph_def = tf.GraphDef()
with open(SAVE_PATH, "rb") as f:
    output_graph_def.ParseFromString(f.read())
    tf.import_graph_def(
        output_graph_def,
        name='',  # 默认name为import,类似scope
    )
sess.run(tf.global_variables_initializer())

in_x = sess.graph.get_tensor_by_name("in_x:0")
keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
prediction = sess.graph.get_tensor_by_name('prediction/prediction:0')
logits = sess.graph.get_tensor_by_name('logits/logits:0')

print('网络加载完毕')


# 1*224*224*3
def predict(img):
    img = img.astype(np.float32)
    img /= 128.
    img -= 1
    prediction_val, logits_val = sess.run(
        [prediction, logits], {
            in_x: img,
            keep_prob: 1.
        }
    )
    return prediction_val, logits_val

predict(np.ones((1,224,224,3), dtype=np.uint8))
print('网络初始化完毕')
