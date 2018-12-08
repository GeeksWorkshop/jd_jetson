# coding=utf-8


import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

learning_rate = .001
TRAIN_STEP = 10000
# 需要使用大样本训练>=128，否则很难收敛
BATCH_SIZE = 64
SHOW_STEP = 100
SHOW_SIZE = 100000

# 输入是点集
in_x = tf.placeholder(tf.float32, (None, 2))
# 输出是分类
in_y = tf.placeholder(tf.float32, (None, 3))


def get_net():
    with slim.arg_scope(
            [slim.fully_connected],
            # activation_fn=tf.nn.relu6,
    ):
        net = slim.fully_connected(in_x, 128)
        net = slim.fully_connected(net, 64)
        net = slim.fully_connected(net, 32)
        net = slim.fully_connected(net, 16)
        net = slim.fully_connected(net, 8)
        net = slim.fully_connected(net, 3)
    return net


net = get_net()
loss = tf.reduce_mean((net - in_y) ** 2)
# 使用交叉熵很难收敛。。。必须使用大样本>=512
# 256也有可能收敛。。。。玄学炼丹。。。
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=in_y, logits=net))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
predict = tf.equal(tf.argmax(net, 1), tf.argmax(in_y, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())


def get_data(batch_size):
    points = np.random.uniform(-3, 3, (batch_size, 2))
    #  默认0 表示两个椭圆之外的区域
    labels = np.full(batch_size, 0.)
    # 交集表示共有区域
    ids = ((points[:, 1] ** 2 + points[:, 0] ** 2 / 4) <= 1) & ((points[:, 1] ** 2) / 4 + points[:, 0] ** 2 <= 1)
    labels[ids] = 1.
    # 异或表示并集减交集
    ids = ((points[:, 1] ** 2 + points[:, 0] ** 2 / 4) <= 1) ^ ((points[:, 1] ** 2) / 4 + points[:, 0] ** 2 <= 1)
    labels[ids] = 2.
    return points, labels


def validation(batch_size):
    points, _ = get_data(batch_size)
    _ = tf.keras.utils.to_categorical(_, 3)
    label, loss_val, accuracy_val = sess.run(
        [net, loss, accuracy], {
            in_x: points / 3,
            in_y: _,
        })
    print(loss_val, accuracy_val)
    label = np.argmax(label, axis=1)

    #ids = label == 0
    # plt.plot(points[ids][:, 0], points[ids][:, 1], 'r.')
    #ids = label == 1
    # plt.plot(points[ids][:, 0], points[ids][:, 1], 'g.')
    # ids = label == 2
    # plt.plot(points[ids][:, 0], points[ids][:, 1], 'b.')
    # plt.show()


def main():
    for i in range(1, TRAIN_STEP + 1):
        points, labels = get_data(BATCH_SIZE)
        labels = tf.keras.utils.to_categorical(labels, 3)
        # 对点集不使用归一化的话也很难收敛
        sess.run(train, {
            in_x: points / 3,
            in_y: labels,
        })

        if not i % SHOW_STEP:
            validation(SHOW_SIZE)


if __name__ == '__main__':
    main()
