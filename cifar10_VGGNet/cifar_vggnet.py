import tensorflow as tf
import cifar10_input
import matplotlib.pyplot as plt
import numpy as np
import os

def get_weight_variable(shape):
    weights = tf.get_variable("weights", shape, dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    return weights

def conv_op(name, input, n_out, is_training):
    with tf.variable_scope(name):
        n_in = input.get_shape()[-1].value
        kernel = get_weight_variable(shape=[3, 3, n_in, n_out])
        tf.summary.histogram('weights', kernel)
        bias = tf.get_variable("biases", shape=[n_out], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))
        tf.summary.histogram('biases', bias)
        conv2d = tf.nn.conv2d(
            input, kernel, strides=[
                1, 1, 1, 1], padding='SAME')
        cache = tf.nn.bias_add(conv2d, bias)
        bn = tf.contrib.layers.batch_norm(cache, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                          is_training=is_training, updates_collections=tf.GraphKeys.UPDATE_OPS)
        activation = tf.nn.relu(bn)
        tf.summary.histogram('active', activation)
        return activation

def maxpool_op(name, tensor):
    with tf.variable_scope(name):
        return tf.nn.max_pool(tensor, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

# 注：bacth_norm函数 is_training:图层是否处于训练模式。在训练模式下，它将积累转入的统计量moving_mean并
# moving_variance使用给定的指数移动平均值 decay。
# 当它不是在训练模式，那么它将使用的数值moving_mean和moving_variance。
# 训练时，需要更新moving_mean和moving_variance。
# 默认情况下，更新操作被放入tf.GraphKeys.UPDATE_OPS，所以需要添加它们作为依赖项train_op。
# 例如：update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
#  with tf.control_dependencies(update_ops):   
# train_op = optimizer.minimize(loss)
# 可以将updates_collections = None设置为强制更新，但可能会导致速度损失，尤其是在分布式设置中。

def fc_op(name, channel, flatten, is_training):
    with tf.variable_scope(name):
        node = flatten.get_shape()[-1].value
        theta = get_weight_variable(shape=[node, channel])
        tf.summary.histogram('weights', theta)
        bias = tf.get_variable("biases", shape=[channel], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        tf.summary.histogram('biases', bias)
        cache = tf.matmul(flatten, theta) + bias
        bn = tf.contrib.layers.batch_norm(cache, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                          is_training=is_training, updates_collections=tf.GraphKeys.UPDATE_OPS)
        fc_connect = tf.nn.relu(bn)
        tf.summary.histogram('fully_connect', fc_connect)
        return fc_connect

def vggnet_inference(image_input, bsize, keep_prob, is_training):
    conv11 = conv_op('conv1_1', image_input, 64, is_training)
    conv12 = conv_op('conv1_2', conv11, 64, is_training)
    pool1 = maxpool_op('pool1', conv12)
    conv21 = conv_op('conv2-1', pool1, 128, is_training)
    conv22 = conv_op('conv2_2', conv21, 128, is_training)
    pool2 = maxpool_op('pool2', conv22)
    conv31 = conv_op('conv3_1', pool2, 256, is_training)
    conv32 = conv_op('conv3_2', conv31, 256, is_training)
    conv33 = conv_op('conv3_3', conv32, 256, is_training)
    pool3 = maxpool_op('pool3', conv33)
    conv41 = conv_op('conv4_1', pool3, 512, is_training)
    conv42 = conv_op('conv4_2', conv41, 512, is_training)
    conv43 = conv_op('conv4_3', conv42, 512, is_training)
    pool4 = maxpool_op('pool4', conv43)
    conv51 = conv_op('conv5_1', pool4, 512, is_training)
    conv52 = conv_op('conv5_2', conv51, 512, is_training)
    conv53 = conv_op('conv5_3', conv52, 512, is_training)
    # pool5 = maxpool_op('pool5', conv53)
    repool5 = tf.reshape(conv53, [bsize, -1])
    fc6 = fc_op('fc6', 4096, repool5, is_training)
    fc61 = tf.nn.dropout(fc6, keep_prob)
    fc7 = fc_op('fc7', 4096, fc61, is_training)
    fc71 = tf.nn.dropout(fc7, keep_prob)
    log = fc_op('fc8', 10, fc71, is_training)
    return log

def loss(labels, prediction, coeff):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=prediction)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    total_loss = cross_entropy_mean + l2 * coeff
    tf.summary.scalar('loss', total_loss)
    return total_loss

def learning_rate_schedule(epoch_num):
    if epoch_num < 81:
        return 0.1
    elif epoch_num < 121:
        return 0.01
    else:
        return 0.001

if __name__ == '__main__':
    batch_size = 128
    max_step = 10000
    lambd = 3e-4
    model_save_path = '/deeplearning_cifar10/model'
    model_name = 'cifar_vggnet.ckpt'

    x_holder = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, 24, 24, 3],
        name='x_input')
    tf.summary.image('input_image', x_holder, 10)
    y_holder = tf.placeholder(
        dtype=tf.int64, shape=[batch_size], name='y_output')
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    global_step = tf.Variable(0, trainable=False)
    logits = vggnet_inference(x_holder, batch_size, keep_prob, is_training)
    cost = loss(y_holder, logits, lambd)
    ops_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(ops_update):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            cost, global_step=global_step)
    correct_prediction = tf.equal(tf.argmax(logits, 1), y_holder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    saver = tf.train.Saver()

    coord = tf.train.Coordinator()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(
            "/deeplearning_cifar10/logs/", sess.graph)
        data_dir = r'E:\cifar10_data\cifar-10-batches-bin'
        train_image, train_label = cifar10_input.distorted_inputs(
            data_dir=data_dir, batch_size=batch_size)
        test_image, test_label = cifar10_input.inputs(
            eval_data=True, data_dir=data_dir, batch_size=batch_size)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        loss_list = []
        for i in range(max_step):
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)#配置运行时需要记录的信息
            # run_metadata = tf.RunMetadata()#运行时记录运行信息的proto
            lr = learning_rate_schedule(max_step)
            op_train, op_labels = sess.run([train_image, train_label])
            _, loss, step = sess.run([train_op, cost, global_step], feed_dict={
                x_holder: op_train, y_holder: op_labels, keep_prob: 0.5, learning_rate: lr, is_training: True})
            precision = sess.run(
                accuracy,
                feed_dict={
                    x_holder: op_train,
                    y_holder: op_labels,
                    keep_prob: 1,
                    is_training: False})
            if step % 10 == 0:
                #writer.add_run_metadata(run_metadata, 'step%03d' % step)
                result = sess.run(
                    merged,
                    feed_dict={
                        x_holder: op_train,
                        y_holder: op_labels, keep_prob: 0.5, learning_rate: lr, is_training: True})
                print(
                    "after {:d} training step ,loss is {:0.3f}, accuracy is {:0.3f}".format(
                        step, loss, precision))
                loss_list.append(loss)
                writer.add_summary(result, step)

            if step % 1000 == 0:
                saver.save(sess,
                           os.path.join(
                               model_save_path,
                               model_name),
                           global_step=global_step)

        ''''
        x1 = np.arange(10, max_step + 10, 10)
        plt.figure()
        plt.plot(x1, loss_list, 'r')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        '''
        writer.close()

        data_image, data_label = sess.run([test_image, test_label])
        test_acc = sess.run(
            accuracy,
            feed_dict={
                x_holder: data_image,
                y_holder: data_label, keep_prob: 1, is_training: False})
        print(
            "after {:d} steps ,test accuracy is {:0.3f}".format(
                max_step, test_acc))
        coord.request_stop()
        coord.join(threads)
