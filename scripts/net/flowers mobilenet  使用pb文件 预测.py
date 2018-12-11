# -*- coding: utf-8 -*-  

import tensorflow as tf
import os
import numpy as np
from PIL import Image
import time

SAVE_DIR = './pb/'
SAVE_PATH = os.path.join(SAVE_DIR, 'flowers_mobilenet.pb')
IMAGE_DIR = './imgs'

sess = tf.Session()
output_graph_def = tf.GraphDef()
with open(SAVE_PATH, "rb") as f:
    output_graph_def.ParseFromString(f.read())
    tf.import_graph_def(
        output_graph_def,
        name='',  # 默认name为import,类似scope
        # return_elements=['prediction:0']
    )
sess.run(tf.global_variables_initializer())

in_x = sess.graph.get_tensor_by_name("in_x:0")
keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')

prediction = sess.graph.get_tensor_by_name('prediction/prediction:0')
logits = sess.graph.get_tensor_by_name('logits/logits:0')
print(in_x.name, prediction.name, logits.name)
print(in_x.shape, prediction.shape, logits.shape)

for name in os.listdir(IMAGE_DIR):
    path = os.path.join(IMAGE_DIR, name)
    img = Image.open(path).resize((224, 224))
    img = np.array(img).astype(np.float32).reshape((-1, 224, 224, 3))
    img /= 128.
    img -= 1
    st = time.time()
    prediction_val, logits_val = sess.run(
        [prediction, logits], {
            in_x: img,
            keep_prob: 1.
        }
    )
    print(name, time.time() - st)
    print(prediction_val, logits_val)
