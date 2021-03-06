import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets( "MNIST_data/", one_hot=True )


#
# -------------------------------------------
#
# Global variables

batch_size = 128
z_dim = 10

#
# ==================================================================
# ==================================================================
# ==================================================================
#

def linear( in_var, output_size, name="linear", stddev=0.02, bias_val=0.0 ):
    shape = in_var.get_shape().as_list()

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", [shape[1], output_size], tf.float32,
                            tf.random_normal_initializer( stddev=stddev ) )
        b = tf.get_variable( "b", [output_size],
                             initializer=tf.constant_initializer( bias_val ))

        return tf.matmul( in_var, W ) + b

def lrelu( x, leak=0.2, name="lrelu" ):
    return tf.maximum( x, leak*x )

def deconv2d( in_var, output_shape, name="deconv2d", stddev=0.02, bias_val=0.0 ):
    k_w = 5  # filter width/height
    k_h = 5
    d_h = 2  # x,y strides
    d_w = 2

    # [ height, width, in_channels, number of filters ]
    var_shape = [ k_w, k_h, output_shape[-1], in_var.get_shape()[-1] ]

    with tf.variable_scope( name ):    
        W = tf.get_variable( "W", var_shape,
                             initializer=tf.truncated_normal_initializer( stddev=0.02 ) )
        b = tf.get_variable( "b", [output_shape[-1]],
                             initializer=tf.constant_initializer( bias_val ))

        deconv = tf.nn.conv2d_transpose( in_var, W, output_shape=output_shape, strides=[1, d_h, d_w, 1] )
        deconv = tf.reshape( tf.nn.bias_add( deconv, b), deconv.get_shape() )
    
        return deconv

def conv2d( in_var, output_dim, name="conv2d" ):
    k_w = 5  # filter width/height
    k_h = 5
    d_h = 2  # x,y strides
    d_w = 2

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", [k_h, k_w, in_var.get_shape()[-1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.02) )
        b = tf.get_variable( "b", [output_dim], initializer=tf.constant_initializer(0.0) )

        conv = tf.nn.conv2d( in_var, W, strides=[1, d_h, d_w, 1], padding='SAME' )
        conv = tf.reshape( tf.nn.bias_add( conv, b ), conv.get_shape() )

        return conv

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# the generator should accept a (tensor of multiple) 'z' and return an image
# z will be [None,z_dim]


#def gen_model(z):
#    g_h1 = linear(z, 128*7*7, name='g_h1')
#    g_h1 = tf.nn.relu(g_h1)
#    g_h1 = tf.reshape(g_h1, [batch_size,7,7,128])
#    g_dc2 = deconv2d(g_h1, [batch_size,14,14,128], name='g_dc2')
#    g_dc2 = tf.nn.relu(g_dc2)
#    g_dc3 = deconv2d(g_dc2, [batch_size,28,28,1], name='g_dc3')
#    g_dc3 = tf.reshape(g_dc3, [batch_size,784])
#    return tf.sigmoid(g_dc3)

def gen_model( batch ):
    H1 = linear(z, 128*7*7, name = 'g_linear_1')
    relu_1 = lrelu(H1, leak = 0, name = 'g_relu_1')
    relu_1 = tf.reshape(relu_1, [batch_size, 7, 7, 128]) #?????????????

    

    D2 = deconv2d( relu_1, [batch_size, 14, 14, 128], name="g_deconv2d_2", stddev=0.02, bias_val=0.0 )
    relu_2 = lrelu(D2, leak = 0, name = 'g_relu_2')

    D3 = deconv2d( relu_2, [batch_size, 28, 28, 1], name="g_deconv2d_3", stddev=0.02, bias_val=0.0 )
    D3 = tf.reshape(D3, [batch_size, 28,28,1])
    return tf.sigmoid(D3)

# -------------------------------------------
    
# the discriminator should accept a (tensor of muliple) images and
# return a probability that the image is real
# imgs will be [None,784]




def disc_model(imgs):
    imgs = tf.reshape( imgs, [batch_size, 28,28,1])
    H0 = conv2d(imgs, 32, name = "d_conv2d_0")
    lr0 = lrelu( H0 , name = "d_leakyrelu_0")

    H1 = conv2d(lr0, 64, name = "d_conv2d_1")
    H1 = tf.reshape(H1, [batch_size, -1])
    lr1 = lrelu( H1 , name = "d_leakyrelu_1")

    H2 = linear( lr1, 1024, name="d_linear_0")

    H3 = linear( H2, 1, name = "d_linear_1")

    return tf.sigmoid(H3)


#
# ==================================================================
# ==================================================================
# ==================================================================
#

# Create your computation graph, cost function, and training steps here!

# Placeholders should be named 'z' and ''true_images'
# Training ops should be named 'd_optim' and 'g_optim'
# The output of the generator should be named 'sample_images'
    
#
# ==================================================================
# ==================================================================
# ==================================================================
#
#Placeholders
z = tf.placeholder(tf.float32, shape = [None,z_dim], name = 'z')
true_images = tf.placeholder(tf.float32, shape = [None, 784], name = 'true_images')

with tf.variable_scope('Graph') as scope:
    sample_images = gen_model(z)
    true_images_probs = disc_model(true_images)
    scope.reuse_variables()
    sample_images_probs = disc_model(sample_images)

d_loss = tf.reduce_mean(-tf.add(tf.log(true_images_probs),tf.log(tf.add(1.,-sample_images_probs))))
g_loss = tf.reduce_mean(-tf.log(sample_images_probs))

d_true_acc = tf.reduce_mean(tf.cast(tf.greater(true_images_probs, 0.5), tf.float32))
d_sample_acc = tf.reduce_mean(tf.cast(tf.less(sample_images_probs, 0.5), tf.float32))
d_acc = tf.truediv(tf.add(d_true_acc, d_sample_acc), 2.0)

t_vars = tf.trainable_variables()
step_size = 0.0002
beta = 0.5

d_vars = [var for var in t_vars if 'd_' in var.name]
d_optim = tf.train.AdamOptimizer( step_size, beta1=beta ).minimize( d_loss, var_list=d_vars )

g_vars = [var for var in t_vars if 'g_' in var.name]
g_optim = tf.train.AdamOptimizer( step_size, beta1=beta ).minimize( g_loss, var_list=g_vars )


sess = tf.Session()
sess.run( tf.initialize_all_variables() )
summary_writer = tf.train.SummaryWriter( "./tf_logs", graph=sess.graph )

for i in range( 500 ):
    batch = mnist.train.next_batch( batch_size )
    batch_images = batch[0]

    sampled_zs = np.random.uniform( low=-1, high=1, size=(batch_size, z_dim) ).astype( np.float32 )
    sess.run( d_optim, feed_dict={ z:sampled_zs, true_images: batch_images } )

    for j in range(3):
        sampled_zs = np.random.uniform( low=-1, high=1, size=(batch_size, z_dim) ).astype( np.float32 )
        sess.run( g_optim, feed_dict={ z:sampled_zs } )
    
    if True:
    #if i%10==0:
        d_acc_val,d_loss_val,g_loss_val = sess.run( [d_acc,d_loss,g_loss],
                                                    feed_dict={ z:sampled_zs, true_images: batch_images } )
        print "%d\t%.2f %.2f %.2f" % ( i, d_loss_val, g_loss_val, d_acc_val )

summary_writer.close()

#
#  show some results
#
sampled_zs = np.random.uniform( -1, 1, size=(batch_size, z_dim) ).astype( np.float32 )
simgs = sess.run( sample_images, feed_dict={ z:sampled_zs } )
simgs = simgs[0:64,:]

tiles = []
for i in range(0,8):
    tiles.append( np.reshape( simgs[i*8:(i+1)*8,:], [28*8,28] ) )
plt.imshow( np.hstack(tiles), interpolation='nearest', cmap=matplotlib.cm.gray )
plt.colorbar()
plt.show()
