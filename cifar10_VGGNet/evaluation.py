import tensorflow as tf
import cifar10_input
import cifar_vggnet


data_dir = r'E:\cifar10_data\cifar-10-batches-bin'
model_save_path = r'\deeplearning_cifar10\model'
batch_size = 256
test_images, test_labels = cifar10_input.inputs(
    eval_data=True, data_dir=data_dir, batch_size=batch_size)
x_holder = tf.placeholder(
    dtype=tf.float32,
    shape=[
        batch_size,
        24,
        24,
        3],
    name='images')
y_holder = tf.placeholder(dtype=tf.int64, shape=[batch_size], name='labels')
is_training = tf.placeholder(dtype=tf.bool)
keeprop = tf.placeholder(dtype=tf.float32)
predictions = cifar_vggnet.vggnet_inference(
    x_holder, batch_size, keeprop, is_training)
acc = tf.nn.in_top_k(predictions, y_holder, 1)
accuracy = tf.reduce_mean(tf.cast(acc, dtype=tf.float32))
coord = tf.train.Coordinator()
saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_save_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        queue = tf.train.start_queue_runners(sess=sess, coord=coord)
        images_op, labels_op = sess.run([test_images, test_labels])
        precision = sess.run(accuracy, feed_dict={x_holder: images_op, y_holder: labels_op, keeprop: 1, is_training: False
                                                  })
        print(
            "after {:s} steps, the precision is {:0.3f}".format(
                global_step,
                precision))

        coord.request_stop()
        coord.join(queue)
