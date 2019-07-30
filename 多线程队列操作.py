import tensorflow as tf
q=tf.FIFOQueue(100,"float32")
enq_op=q.enqueue([tf.random_normal([1])])
qr=tf.train.QueueRunner(q,[enq_op]*5)
tf.train.add_queue_runner(qr)
out_tensor=q.dequeue()
with tf.Session() as sess:
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    for _ in range(3):
        print(sess.run(out_tensor)[0])
    coord.request_stop()
    coord.join(threads)
