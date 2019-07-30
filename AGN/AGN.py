import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(n_in,n_out):
    low=-np.sqrt(6.0/(n_in+n_out))
    high=np.sqrt(6.0/(n_in+n_out))
    return tf.random_uniform([n_in,n_out],minval=low,maxval=high,dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder():
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.transfer=transfer_function
        self.scale=tf.placeholder(tf.float32)
        self.training_scale=scale
        self.weights=self._initialize_weights_()

        self.x=tf.placeholder(tf.float32,[None,self.n_input])
        #编码
        self.hidden=self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal([self.n_input,]),self.weights['w1']),self.weights['b1']))
        #解码
        self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
        #cost function
        self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        self.optimizer=optimizer.minimize(self.cost)

        self.sess=tf.Session()
        init=tf.global_variables_initializer()
        self.sess.run(init)

    def _initialize_weights_(self):
        all_weights=dict()
        all_weights['w1']=tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1']=tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2']=tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
        all_weights['b2']=tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights

    def partial_fit(self,X):
        cost,opt=self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X,self.scale:self.training_scale})
        return cost

    def calc_total_cost(self,X):
        cost=self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})
        return cost

    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})

    def generate(self,hidden=None):
        if hidden is None:
            hidden=np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})

    def reconstruction(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

#预处理
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
def standard_scale(X_train,X_test):
    preprocessor=prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train,X_test

def get_random_block_from_data(data,batch_size):
    start_index=np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

#training
X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)
n_samples=int(mnist.train.num_examples)
batch_size=128
training_epochs=20
display_step=1

autoencoder=AdditiveGaussianNoiseAutoencoder(n_input=784,
                                             n_hidden=200,
                                             transfer_function=tf.nn.softplus,
                                             optimizer=tf.train.AdamOptimizer(0.001),
                                             scale=0.01)
for epoch in range(training_epochs+1):
    avg_cost = 0.0
    for i in range(int(n_samples/batch_size)):
        batch_xs=get_random_block_from_data(X_train,batch_size)
        cost=autoencoder.partial_fit(batch_xs)
        avg_cost+=cost/n_samples*batch_size
    if epoch%display_step==0:
        print("epoch:%d,cost:%g"%(epoch,avg_cost))
print("total cost:"+str(autoencoder.calc_total_cost(X_test)))
