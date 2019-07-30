import tensorflow as tf
import threading
import numpy as np
import time
'''
q=tf.FIFOQueue(2,"int32")
init=q.enqueue_many(([0,10],))
x=q.dequeue()
y=x+1
q_inc=q.enqueue([y])
with tf.Session() as sess:
    sess.run(init)
    for i in range(5):
        v,_=sess.run([x,q_inc])
        print(v)
'''
def MyLoop(coord,worker_id):
    while not coord.should_stop():
        if np.random.rand()<0.1:
            print("stop from id:%d\n"%worker_id)
            coord.request_stop()
        else:
            print("work on id:%d\n"%worker_id)
        time.sleep(1)
coord=tf.train.Coordinator()
threads=[threading.Thread(target=MyLoop,args=(coord,i)) for i in range(5)]
for t in threads:t.start()
coord.join(threads)

        

