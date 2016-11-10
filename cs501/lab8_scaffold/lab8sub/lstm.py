#Chris Rytting
import tensorflow as tf
import numpy as np

from tensorflow.python.ops.rnn_cell import BasicLSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops.rnn_cell import _linear

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

#from tensorflow.python.util import nest

from textloader import TextLoader

 
class mygru( RNNCell ):
 
    def __init__( self, num_units, activation = tanh ):
        self._num_units = num_units
        self._activation = activation
 
    @property
    def state_size(self):
        return self._num_units
 
    @property
    def output_size(self):
        return self._num_units
 
    def __call__( self, inputs, state, scope=None ):
        with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            with vs.variable_scope('rz'):
                h = state
                concat = _linear([inputs, h], 2 * self._num_units, False)
                r,z = array_ops.split(1,2, concat)
                r = sigmoid(r)
                z = sigmoid(z)
            with vs.variable_scope('h_tilde'):
                h_tilde = self._activation(_linear([inputs, r * h], self._num_units, False))
                new_h = z * h + (1-z) * h_tilde

        return new_h, new_h

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            #i, j, f, o = array_ops.split(1, 4, concat)

            #new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
            #       self._activation(j))
            #new_h = self._activation(new_c) * sigmoid(o)

            #if self._state_is_tuple:
            #new_state = LSTMStateTuple(new_c, new_h)
            #else:
            #new_state = array_ops.concat(1, [new_c, new_h])
            #return new_h, new_state



#
# -------------------------------------------
#
# Global variables

batch_size = 10
sequence_length = 10

data_loader = TextLoader( ".", batch_size, sequence_length )

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 128

num_layers = 2

tf.reset_default_graph()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# define placeholders for our inputs.  
# in_ph is assumed to be [batch_size,sequence_length]
# targ_ph is assumed to be [batch_size,sequence_length]

in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='inputs' )
targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )

inputs = tf.split( 1, sequence_length, in_onehot )
inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
targets = tf.split( 1, sequence_length, targ_ph )

# at this point, inputs is a list of length sequence_length
# each element of inputs is [batch_size,vocab_size]

# targets is a list of length sequence_length
# each element of targets is a 1D vector of length batch_size

# ------------------
# YOUR COMPUTATION GRAPH HERE

# create a BasicLSTMCell
#   use it to create a MultiRNNCell
#   use it to create an initial_state
#     note that initial_state will be a *list* of tensors!

# call seq2seq.rnn_decoder

# transform the list of state outputs to a list of logits.
# use a linear transformation.

# call seq2seq.sequence_loss

# create a training op using the Adam optimizer

cell1 = mygru( state_dim )
cell2 = mygru( state_dim )

#cell1 = BasicLSTMCell( state_dim, state_is_tuple=True)
#cell2 = BasicLSTMCell( state_dim, state_is_tuple=True)
# Initial state of the LSTM memory.
multicell = MultiRNNCell( [cell1,cell2], state_is_tuple=True)
initial_state = multicell.zero_state(batch_size, tf.float32)

print multicell

rnn_out, final_state = seq2seq.rnn_decoder(inputs, initial_state, multicell)
logits = []
W = tf.Variable(tf.truncated_normal([state_dim, vocab_size]), dtype = tf.float32)
b = tf.Variable(tf.ones(vocab_size)*0.1, dtype = tf.float32)
for rnn in rnn_out:
    logit = tf.matmul(rnn, W) + b
    logits.append(logit)

ones_list = []
for i in xrange(batch_size):
    ones_list.append(1.)
loss = seq2seq.sequence_loss(logits, targets, ones_list)

optim = tf.train.AdamOptimizer(.001).minimize(loss)


# ------------------
# YOUR SAMPLER GRAPH HERE

# place your sampler graph here it will look a lot like your
# computation graph, except with a "batch_size" of 1.

# remember, we want to reuse the parameters of the cell and whatever
# parameters you used to transform state outputs to logits!
tf.get_variable_scope().reuse_variables()



s_initial_state = multicell.zero_state(1, tf.float32)
s_in_ph = tf.placeholder( tf.int32, [ 1 ], name='s_in_ph' )
s_inputs = tf.one_hot( s_in_ph, vocab_size, name="s_inputs" )
s_rnn_out, s_final_state = seq2seq.rnn_decoder([s_inputs,], s_initial_state, multicell)

s_logits = tf.matmul(s_rnn_out[0], W) + b
s_probs = tf.nn.softmax(s_logits)


#
# ==================================================================
# ==================================================================
# ==================================================================
#





def sample( num=200, prime='ab' ):

    # prime the pump 

    # generate an initial state. this will be a list of states, one for
    # each layer in the multicell.
    s_state = sess.run( s_initial_state )

    # for each character, feed it into the sampler graph and
    # update the state.
    for char in prime[:-1]:
        x = np.ravel( data_loader.vocab[char] ).astype('int32')
        feed = { s_in_ph:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        s_state = sess.run( s_final_state, feed_dict=feed )

    # now we have a primed state vector; we need to start sampling.
    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel( data_loader.vocab[char] ).astype('int32')

        # plug the most recent character in...
        feed = { s_in_ph:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend( list(s_final_state) )

        retval = sess.run( ops, feed_dict=feed )

        s_probsv = retval[0]
        s_state = retval[1:]

        # ...and get a vector of probabilities out!

        # now sample (or pick the argmax)
        # sample = np.argmax( s_probsv[0] )
        sample = np.random.choice( vocab_size, p=s_probsv[0] )

        pred = data_loader.chars[sample]
        ret += pred
        char = pred

    return ret

#
# ==================================================================
# ==================================================================
# ==================================================================
#

sess = tf.Session()
sess.run( tf.initialize_all_variables() )
summary_writer = tf.train.SummaryWriter( "./tf_logs", graph=sess.graph )

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches

for j in range(1000):

    state = sess.run( initial_state )
    data_loader.reset_batch_pointer()

    for i in range( data_loader.num_batches ):
        
        x,y = data_loader.next_batch()

        # we have to feed in the individual states of the MultiRNN cell
        feed = { in_ph: x, targ_ph: y }
        for k, s in enumerate( initial_state ):
            feed[s] = state[k]

        ops = [optim,loss]
        ops.extend( list(final_state) )

        # retval will have at least 3 entries:
        # 0 is None (triggered by the optim op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        retval = sess.run( ops, feed_dict=feed )

        lt = retval[1]
        state = retval[2:]

        if i%1000==0:
            print "%d %d\t%.4f" % ( j, i, lt )
            lts.append( lt )

    print sample( num=60, prime="I " )
#    print sample( num=60, prime="And " )
#    print sample( num=60, prime="ababab" )
#    print sample( num=60, prime="foo ba" )
#    print sample( num=60, prime="abcdab" )

summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

#import matplotlib
#import matplotlib.pyplot as plt
#plt.plot( lts )
#plt.show()
