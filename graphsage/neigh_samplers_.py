from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import numpy as np
import pdb
"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        
        # retrieve matrix of [numofids, degree(128)]
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        # shuffling along degree axis 
        #adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        # pick [numofids, num_samples]
        #adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
       
        unif_rand = tf.random.uniform([num_samples], minval=0, maxval=np.int(adj_lists.shape[1]), dtype=tf.int32)
        
        adj_lists = tf.gather(adj_lists, unif_rand, axis=1)
        

        condition = tf.equal(adj_lists, self.adj_info.shape[0]-1)
        case_true = tf.zeros(adj_lists.shape, tf.float32)
        case_false = tf.ones(adj_lists.shape, tf.float32)
        adj_lists_numnz = tf.reduce_sum(tf.where(condition, case_true, case_false), axis=1)
 
        att_lists = tf.ones(adj_lists.shape)
        dummy_ = adj_lists_numnz

        return adj_lists, att_lists, adj_lists_numnz, dummy_
        #return adj_lists


class MLNeighborSampler(Layer):

    def __init__(self, adj_info, features, **kwargs):
        super(MLNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
        self.batch_size = FLAGS.max_degree
        self.node_dim = features.shape[1]
        self.reuse = False 

    def _call(self, inputs):
       
        ids, num_samples = inputs

        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        vert_num = ids.shape[0]
        neig_num = self.adj_info.shape[1]

        # build model 
        # l = W*x1
        # l = relu(l*x2^t)
        with tf.variable_scope("MLsampler"):

           
            v_f = tf.nn.embedding_lookup(self.features, ids)
            n_f = tf.nn.embedding_lookup(self.features, tf.reshape(adj_lists, [-1]))
           

            n_f = tf.reshape(n_f, shape=[-1, neig_num, self.node_dim])

            
            if FLAGS.nonlinear_sampler == True:
            
                v_f = tf.tile(tf.expand_dims(v_f, axis=1), [1, neig_num, 1])
           
                l = tf.layers.dense(tf.concat([v_f, n_f], axis=2), 1, activation=None, trainable=False, reuse=self.reuse, name='dense')
           
                out = tf.nn.relu(tf.exp(l), name='relu')


            else:

                l = tf.layers.dense(v_f, self.node_dim, activation=None, trainable=False, reuse=self.reuse, name='dense')
           
                l = tf.expand_dims(l, axis=1)
                l = tf.matmul(l, n_f, transpose_b=True, name='matmul') 
           
                out = tf.nn.relu(l, name='relu')

            
            out = tf.squeeze(out)

            '''
            condition = tf.equal(out, 0.0)
            case_true = tf.multiply(tf.ones(out.shape, tf.float32), 9999)
            case_false = out
            out = tf.where(condition, case_true, case_false)
            '''

            # sort (sort top k of negative estimated loss)
            out, idx_y = tf.nn.top_k(-out, num_samples)
            idx_y = tf.cast(idx_y, tf.int32)
           
            idx_x = tf.range(vert_num)
            idx_x = tf.tile(tf.expand_dims(idx_x, -1), [1, num_samples])

            adj_lists = tf.gather_nd(adj_lists, tf.stack([tf.reshape(idx_x,[-1]), tf.reshape(idx_y,[-1])], axis=1))
            
            adj_lists = tf.reshape(adj_lists, [vert_num, num_samples])
            
            condition = tf.equal(adj_lists, self.adj_info.shape[0]-1)
            case_true = tf.zeros(adj_lists.shape, tf.float32)
            case_false = tf.ones(adj_lists.shape, tf.float32)
            adj_lists_numnz = tf.reduce_sum(tf.where(condition, case_true, case_false), axis=1)
           
            #out = tf.exp(out)
            #norm = tf.tile(tf.expand_dims(tf.reduce_sum(out, axis=1), -1), [1, num_samples])
            #att = tf.div(out, norm)
            
            #att = tf.nn.softmax(out,axis=1)
            #att_lists = tf.reshape(1+att, [vert_num, num_samples])  
            att_lists = tf.ones(adj_lists.shape)
            dummy_ = adj_lists_numnz            
            
            self.reuse = True

        #return adj_lists
        return adj_lists, att_lists, adj_lists_numnz, dummy_


class FastMLNeighborSampler(Layer):

    def __init__(self, adj_info, features, **kwargs):
        super(FastMLNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
        self.batch_size = FLAGS.max_degree
        self.node_dim = features.shape[1]
        self.reuse = False 

    def _call(self, inputs):
       
        ids, num_samples = inputs

        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        neig_num = np.int(self.adj_info.shape[1])
        
        #unif_rand = tf.random.uniform([np.int(neig_num/num_samples)*num_samples], minval=0, maxval=np.int(neig_num), dtype=tf.int32)
        #adj_lists = tf.gather(adj_lists, unif_rand, axis=1)
        
        
        #neig_num = 0.6*neig_num
        adj_lists = tf.slice(adj_lists, [0,0], [-1, np.int(neig_num/num_samples)*num_samples])
        

        #adj_lists = tf.slice(adj_lists, [0,0], [-1, np.int(neig_num/num_samples)*num_samples])


        vert_num = np.int(adj_lists.shape[0])
        neig_num = np.int(adj_lists.shape[1])
 
        # build model 
        # l = W*x1
        # l = relu(l*x2^t)
        with tf.variable_scope("MLsampler"):

           
            v_f = tf.nn.embedding_lookup(self.features, ids)
            n_f = tf.nn.embedding_lookup(self.features, tf.reshape(adj_lists, [-1]))
            
            # debug
            v_f = tf.slice(v_f, [0,0], [-1,50])
            n_f = tf.slice(n_f, [0,0], [-1,50])

            n_f = tf.reshape(n_f, shape=[-1, neig_num, 50])
            #n_f = tf.reshape(n_f, shape=[-1, neig_num, self.node_dim])

            
            if FLAGS.nonlinear_sampler == True:
            
                v_f = tf.tile(tf.expand_dims(v_f, axis=1), [1, neig_num, 1])
           
                l = tf.layers.dense(tf.concat([v_f, n_f], axis=2), 1, activation=None, trainable=False, reuse=self.reuse, name='dense')
           
                out = tf.nn.relu(tf.exp(l), name='relu')


            else:

                l = tf.layers.dense(v_f, self.node_dim, activation=None, trainable=False, reuse=self.reuse, name='dense')
           
                l = tf.expand_dims(l, axis=1)
                l = tf.matmul(l, n_f, transpose_b=True, name='matmul') 
           
                out = tf.nn.relu(l, name='relu')

            
            out = tf.squeeze(out)

            '''
            condition = tf.equal(out, 0.0)
            case_true = tf.multiply(tf.ones(out.shape, tf.float32), 9999)
            case_false = out
            out = tf.where(condition, case_true, case_false)
            '''

            # sort (sort top k of negative estimated loss)
            #out, idx_y = tf.nn.top_k(-out, num_samples)
            #idx_y = tf.cast(idx_y, tf.int32)
           
            
            # group min
            group_dim = np.int(neig_num/num_samples)
            out = tf.reshape(out, [vert_num, num_samples, group_dim])
            idx_y = tf.argmin(out, axis=-1, output_type=tf.int32)
            #idx_y = tf.squeeze(tf.nn.top_k(-out, k=1)[1])
            delta = tf.expand_dims(tf.range(0, group_dim*num_samples, group_dim), axis=0)
            delta = tf.tile(delta, [vert_num, 1])
            idx_y = idx_y + delta


            idx_x = tf.range(vert_num)
            idx_x = tf.tile(tf.expand_dims(idx_x, -1), [1, num_samples])

            adj_lists = tf.gather_nd(adj_lists, tf.stack([tf.reshape(idx_x,[-1]), tf.reshape(idx_y,[-1])], axis=1))
            adj_lists = tf.reshape(adj_lists, [vert_num, num_samples])
            
            condition = tf.equal(adj_lists, self.adj_info.shape[0]-1)
            case_true = tf.zeros(adj_lists.shape, tf.float32)
            case_false = tf.ones(adj_lists.shape, tf.float32)
            adj_lists_numnz = tf.reduce_sum(tf.where(condition, case_true, case_false), axis=1)
           
            #out = tf.exp(out)
            #norm = tf.tile(tf.expand_dims(tf.reduce_sum(out, axis=1), -1), [1, num_samples])
            #att = tf.div(out, norm)
            
            #att = tf.nn.softmax(out,axis=1)
            #att_lists = tf.reshape(1+att, [vert_num, num_samples])  
            att_lists = tf.ones(adj_lists.shape)
            dummy_ = adj_lists_numnz            
            
            self.reuse = True

        #return adj_lists
        return adj_lists, att_lists, adj_lists_numnz, dummy_


class FastMLNeighborSampler__(Layer):

    def __init__(self, adj_info, features, **kwargs):
        super(FastMLNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
        self.batch_size = FLAGS.max_degree
        self.node_dim = features.shape[1]
        self.reuse = False 

    def _call(self, inputs):
       
        ids, num_samples = inputs

        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        
       
        unif_rand = tf.random.uniform([num_samples], minval=0, maxval=np.int(adj_lists.shape[1]), dtype=tf.int32)
        
        adj_lists = tf.gather(adj_lists, unif_rand, axis=1)
        
        vert_num = np.int(adj_lists.shape[0])
        neig_num = np.int(adj_lists.shape[1])
 
        # build model 
        # l = W*x1
        # l = relu(l*x2^t)
        with tf.variable_scope("MLsampler"):

           
            v_f = tf.nn.embedding_lookup(self.features, ids)
            n_f = tf.nn.embedding_lookup(self.features, tf.reshape(adj_lists, [-1]))
           

            n_f = tf.reshape(n_f, shape=[-1, neig_num, self.node_dim])

            
            if FLAGS.nonlinear_sampler == True:
            
                v_f = tf.tile(tf.expand_dims(v_f, axis=1), [1, neig_num, 1])
           
                l = tf.layers.dense(tf.concat([v_f, n_f], axis=2), 1, activation=None, trainable=False, reuse=self.reuse, name='dense')
           
                out = tf.nn.relu(tf.exp(l), name='relu')


            else:

                l = tf.layers.dense(v_f, self.node_dim, activation=None, trainable=False, reuse=self.reuse, name='dense')
           
                l = tf.expand_dims(l, axis=1)
                l = tf.matmul(l, n_f, transpose_b=True, name='matmul') 
           
                out = tf.nn.relu(l, name='relu')

            
            out = tf.squeeze(out)
           
            '''
            out_sum = tf.reduce_sum(out, axis=1)
            out_nzero = tf.count_nonzero(out, axis=1)
            out_mean = tf.div(out_sum, tf.cast(out_nzero, tf.float32))

            condition = tf.equal(out, 0.0)
            case_true = tf.multiply(tf.ones(out.shape, tf.float32), 99999999)
            case_false = out
            out = tf.where(condition, case_true, case_false)
            '''
            out_mean = tf.reduce_mean(out, axis=1)

            condition = tf.logical_or(tf.equal(adj_lists, self.adj_info.shape[0]-1), tf.greater(out, tf.tile(tf.expand_dims(0.5*out_mean, -1), [1, neig_num])))
            
            case_true0 = (self.adj_info.shape[0]-1)*tf.ones(adj_lists.shape, tf.int32) 
            case_false0 = adj_lists
            
            case_true = tf.zeros(adj_lists.shape, tf.float32)
            case_false = tf.ones(adj_lists.shape, tf.float32)
            adj_lists = tf.where(condition, case_true0, case_false0)
            adj_lists_numnz = tf.reduce_sum(tf.where(condition, case_true, case_false), axis=1)
            ''' 
            condition = tf.greater(tf.tile(tf.expand_dims(out_mean, -1), [1, neig_num]), out)
            case_true = tf.ones(adj_lists.shape)
            case_false = tf.zeros(adj_lists.shape)
            att_lists = tf.where(condition, case_true, case_false)
            '''

            #att = tf.nn.softmax(-out,axis=1)
            #att_lists = tf.reshape(1+att, [vert_num, num_samples])  
            
            #adj_lists_numnz = tf.count_nonzero(att_lists, axis=1, keepdims=True, dtype=tf.int32)


            #v0 = tf.Variable(vert_num)
            #i0 = tf.constant(0, dtype=tf.int32)
            #m0 = tf.zeros([1, num_samples], dtype=tf.int32)
            #c = lambda i, m, num: i < num
            #pdb.set_trace()
            #b = lambda i, m, num: [i+1, tf.concat([m,tf.cast(tf.transpose(tf.where(condition[i,:])[:num_samples]), tf.int32)], axis=0), num]
            #loop_out = tf.while_loop(c, b, loop_vars=[i0, m0, v0], shape_invariants=[i0.get_shape(), tf.TensorShape([None, num_samples]), v0.get_shape()])
            #idx_y = loop_out[1]


            #idx_y = tf.where(condition)
            #idx_y = tf.transpose(tf.random_shuffle(tf.transpose(idx_y)))
            # pick [numofids, num_samples]
            #idx_y = tf.cast(tf.slice(idx_y, [0,0], [-1, num_samples]), tf.int32)
 
            # sort (sort top k of negative estimated loss)
            #out, idx_y = tf.nn.top_k(-out, num_samples)
            #idx_y = tf.cast(idx_y, tf.int32)
           
            
            
            #idx_x = tf.range(vert_num)
            #idx_x = tf.tile(tf.expand_dims(idx_x, -1), [1, num_samples])

            #adj_lists = tf.gather_nd(adj_lists, tf.stack([tf.reshape(idx_x,[-1]), tf.reshape(idx_y,[-1])], axis=1))
            #
            #adj_lists = tf.reshape(adj_lists, [vert_num, num_samples])
            #
            #condition = tf.equal(adj_lists, self.adj_info.shape[0]-1)
            #case_true = tf.zeros(adj_lists.shape, tf.float32)
            #case_false = tf.ones(adj_lists.shape, tf.float32)
            #adj_lists_numnz = tf.reduce_sum(tf.where(condition, case_true, case_false), axis=1)
           
            #out = tf.exp(out)
            #norm = tf.tile(tf.expand_dims(tf.reduce_sum(out, axis=1), -1), [1, num_samples])
            #att = tf.div(out, norm)
            
            #att = tf.nn.softmax(out,axis=1)
            #att_lists = tf.reshape(1+att, [vert_num, num_samples])  
            att_lists = tf.ones(adj_lists.shape)
            self.reuse = True

        #return adj_lists
        return adj_lists, att_lists, adj_lists_numnz, out_mean




class FastMLNeighborSampler_(Layer):

    def __init__(self, adj_info, features, **kwargs):
        super(FastMLNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
        self.batch_size = FLAGS.max_degree
        self.node_dim = features.shape[1]
        self.reuse = False 

    def _call(self, inputs):
 
        ids, num_samples = inputs
        
        adj_dim = self.adj_info.get_shape().as_list()
        num_samples_c = np.int(adj_dim[1]*FLAGS.uniform_ratio)

        ## retrieve matrix of [numofids, degree(128)]
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        ## shuffling along degree axis 
        #adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        ## pick [numofids, num_samples]
        #adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples_c])
        
        unif_rand = tf.random.uniform([num_samples_c], minval=0, maxval=np.int(adj_lists.shape[1]), dtype=tf.int32)
        
        adj_lists = tf.gather(adj_lists, unif_rand, axis=1)
 
        #vert_num = np.int(adj_lists.shape[0])
        #neig_num = np.int(adj_lists.shape[1])

        #unif_rand = tf.random.uniform([num_samples_c], minval=0, maxval=neig_num, dtype=tf.int32)
        #
        #adj_lists = tf.gather(adj_lists, unif_rand, axis=1)
         

        vert_num = np.int(adj_lists.shape[0])
        neig_num = np.int(adj_lists.shape[1])

        # build model 
        # l = W*x1
        # l = relu(l*x2^t)
        with tf.variable_scope("MLsampler"):

           
            v_f = tf.nn.embedding_lookup(self.features, ids)
            n_f = tf.nn.embedding_lookup(self.features, tf.reshape(adj_lists, [-1]))
           

            n_f = tf.reshape(n_f, shape=[-1, neig_num, self.node_dim])

            
            if FLAGS.nonlinear_sampler == True:
            
                v_f = tf.tile(tf.expand_dims(v_f, axis=1), [1, neig_num, 1])
           
                l = tf.layers.dense(tf.concat([v_f, n_f], axis=2), 1, activation=None, trainable=False, reuse=self.reuse, name='dense')
           
                out = tf.nn.relu(tf.exp(l), name='relu')


            else:

                l = tf.layers.dense(v_f, self.node_dim, activation=None, trainable=False, reuse=self.reuse, name='dense')
           
                l = tf.expand_dims(l, axis=1)
                l = tf.matmul(l, n_f, transpose_b=True, name='matmul') 
           
                out = tf.nn.relu(l, name='relu')

            
            out = tf.squeeze(out)


            condition = tf.equal(out, 0.0)
            case_true = tf.multiply(tf.ones(out.shape, tf.float32), 9999)
            case_false = out
            out = tf.where(condition, case_true, case_false)
    
            # sort (sort top k of negative estimated loss)
            out, idx_y = tf.nn.top_k(-out, num_samples)
            idx_y = tf.cast(idx_y, tf.int32)
           
            
            idx_x = tf.range(vert_num)
            idx_x = tf.tile(tf.expand_dims(idx_x, -1), [1, num_samples])

            adj_lists = tf.gather_nd(adj_lists, tf.stack([tf.reshape(idx_x,[-1]), tf.reshape(idx_y,[-1])], axis=1))
            
            adj_lists = tf.reshape(adj_lists, [vert_num, num_samples])
         
            condition = tf.equal(adj_lists, self.adj_info.shape[0]-1)
            case_true = tf.zeros(adj_lists.shape, tf.float32)
            case_false = tf.ones(adj_lists.shape, tf.float32)
            adj_lists_numnz = tf.reduce_sum(tf.where(condition, case_true, case_false), axis=1)
           
            #out = tf.exp(out)
            #norm = tf.tile(tf.expand_dims(tf.reduce_sum(out, axis=1), -1), [1, num_samples])
            #att = tf.div(out, norm)
            
            #att = tf.nn.softmax(out,axis=1)
            #att_lists = tf.reshape(1+att, [vert_num, num_samples]) 
            att_lists=tf.ones(adj_lists.shape) 

            self.reuse = True

        #return adj_lists
        return adj_lists, att_lists, adj_lists_numnz


#class FastMLNeighborSampler(Layer):
#
#    def __init__(self, adj_info, features, **kwargs):
#        super(FastMLNeighborSampler, self).__init__(**kwargs)
#        self.adj_info = adj_info
#        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
#        self.batch_size = FLAGS.max_degree
#        self.node_dim = features.shape[1]
#        self.reuse = False 
#
#    def _call(self, inputs):
# 
#        ids, num_samples = inputs
#        
#        adj_dim = self.adj_info.get_shape().as_list()
#        #num_samples_c = np.int(adj_dim[1]*FLAGS.uniform_ratio)
#
#        ## retrieve matrix of [numofids, degree(128)]
#        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
#        ## shuffling along degree axis 
#        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
#        ## pick [numofids, num_samples]
#        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
#        
#        #vert_num = np.int(adj_lists.shape[0])
#        #neig_num = np.int(adj_lists.shape[1])
#
#        #unif_rand = tf.random.uniform([num_samples_c], minval=0, maxval=neig_num, dtype=tf.int32)
#        
#        #adj_lists = tf.gather(adj_lists, unif_rand, axis=1)
#         
#
#        vert_num = np.int(adj_lists.shape[0])
#        neig_num = np.int(adj_lists.shape[1])
#
#        # build model 
#        # l = W*x1
#        # l = relu(l*x2^t)
#        with tf.variable_scope("MLsampler"):
#
#           
#            v_f = tf.nn.embedding_lookup(self.features, ids)
#            n_f = tf.nn.embedding_lookup(self.features, tf.reshape(adj_lists, [-1]))
#           
#
#            n_f = tf.reshape(n_f, shape=[-1, neig_num, self.node_dim])
#
#            
#            if FLAGS.nonlinear_sampler == True:
#            
#                v_f = tf.tile(tf.expand_dims(v_f, axis=1), [1, neig_num, 1])
#           
#                l = tf.layers.dense(tf.concat([v_f, n_f], axis=2), 1, activation=None, trainable=False, reuse=self.reuse, name='dense')
#           
#                out = tf.nn.relu(tf.exp(l), name='relu')
#
#
#            else:
#
#                l = tf.layers.dense(v_f, self.node_dim, activation=None, trainable=False, reuse=self.reuse, name='dense')
#           
#                l = tf.expand_dims(l, axis=1)
#                l = tf.matmul(l, n_f, transpose_b=True, name='matmul') 
#           
#                out = tf.nn.relu(l, name='relu')
#
#            
#            out = tf.squeeze(out)
#
#
#            condition = tf.equal(out, 0.0)
#            case_true = tf.multiply(tf.ones(out.shape, tf.float32), 9999)
#            case_false = out
#            out = tf.where(condition, case_true, case_false)
#    
#            # sort (sort top k of negative estimated loss)
##            out, idx_y = tf.nn.top_k(-out, num_samples)
##            idx_y = tf.cast(idx_y, tf.int32)
#           
#           # x_ = np.zeros([vert_num, num_samples])
#           # for j in range(vert_num):
#           #     x_[j,:] = j*np.ones([1, num_samples])
#           #idx_x = tf.Variable(x_, trainable=False, dtype=tf.int32)
#            
#            
##            idx_x = tf.range(vert_num)
##            idx_x = tf.tile(tf.expand_dims(idx_x, -1), [1, num_samples])
#
#            #adj_ids = tf.nn.embedding_lookup(self.adj_info, ids)
#            #adj_ids = adj_lists
##            adj_lists = tf.gather_nd(adj_lists, tf.stack([tf.reshape(idx_x,[-1]), tf.reshape(idx_y,[-1])], axis=1))
#            
##            adj_lists = tf.reshape(adj_lists, [vert_num, num_samples])
#         
#            condition = tf.equal(adj_lists, self.adj_info.shape[0]-1)
#            case_true = tf.zeros(adj_lists.shape, tf.float32)
#            case_false = tf.ones(adj_lists.shape, tf.float32)
#            adj_lists_numnz = tf.reduce_sum(tf.where(condition, case_true, case_false), axis=1)
#           
#            #out = tf.exp(out)
#            #norm = tf.tile(tf.expand_dims(tf.reduce_sum(out, axis=1), -1), [1, num_samples])
#            #att = tf.div(out, norm)
#            
#            att = tf.nn.softmax(-out, axis=1)
#            att_lists = tf.reshape(1+att, [vert_num, neig_num]) 
#            #att_lists=tf.ones(adj_lists.shape) 
#
#            self.reuse = True
#
#        #return adj_lists
#        return adj_lists, att_lists, adj_lists_numnz

#        # build model 
#        # l = W*x1
#        # l = relu(l*x2^t)
#        with tf.variable_scope("MLsampler"):
#
#            import pdb
#            pdb.set_trace()
#            v_f = tf.nn.embedding_lookup(self.features, ids)
#            n_f = tf.nn.embedding_lookup(self.features, tf.reshape(adj_lists, [-1]))
#           
#
#            n_f = tf.reshape(n_f, shape=[-1, neig_num, self.node_dim])
#
#            
#            l = tf.layers.dense(v_f, self.node_dim, activation=None, trainable=False, reuse=self.reuse, name='dense')
#           
#            m = tf.layers.dense(n_f, self.node_dim, activation=None, trainable=False, reuse=self.reuse, name='dense2')
#            
#            l = tf.expand_dims(l, axis=1)
#
#            l = tf.matmul(l, m, transpose_b=True, name='matmul') 
#           
#            out = tf.nn.relu(l, name='relu')
#            out = tf.squeeze(out)
#
#            condition = tf.equal(out, 0.0)
#            case_true = tf.multiply(tf.ones(out.shape, tf.float32), 9999)
#            case_false = out
#            out = tf.where(condition, case_true, case_false)
#    
#            # sort (sort top k of negative estimated loss)
#            out, idx_y = tf.nn.top_k(-out, num_samples)
#            idx_y = tf.cast(idx_y, tf.int32)
#           
#            x_ = np.zeros([vert_num, num_samples])
#            for j in range(vert_num):
#                x_[j,:] = j*np.ones([1, num_samples])
#            
#            idx_x = tf.Variable(x_, trainable=False, dtype=tf.int32)
#            adj_ids = tf.nn.embedding_lookup(self.adj_info, ids)
#            adj_lists = tf.gather_nd(adj_ids, tf.stack([tf.reshape(idx_x,[-1]), tf.reshape(idx_y,[-1])], axis=1))
#            
#            adj_lists = tf.reshape(adj_lists, [vert_num, num_samples])
#
#            self.reuse = True


        
#        # build model 
#        # l = W*x1
#        # l = relu(l*x2^t)
#        with tf.variable_scope("MLsampler"):
#
#           
#            v_f = tf.nn.embedding_lookup(self.features, ids)
#            n_f = tf.nn.embedding_lookup(self.features, tf.reshape(adj_lists, [-1]))
#           
#
#            n_f = tf.reshape(n_f, shape=[-1, neig_num, self.node_dim])
#
#            l = tf.layers.dense(v_f, self.node_dim, activation=None, trainable=False, reuse=self.reuse, name='dense')
#           
#            l = tf.expand_dims(l, axis=1)
#            l = tf.matmul(l, n_f, transpose_b=True, name='matmul') 
#           
#            out = tf.nn.relu(l, name='relu')
#            out = tf.squeeze(out)
#
#            condition = tf.equal(out, 0.0)
#            case_true = tf.multiply(tf.ones(out.shape, tf.float32), 9999)
#            case_false = out
#            out = tf.where(condition, case_true, case_false)
#    
#            # sort (sort top k of negative estimated loss)
#            out, idx_y = tf.nn.top_k(-out, num_samples)
#            idx_y = tf.cast(idx_y, tf.int32)
#           
#            x_ = np.zeros([vert_num, num_samples])
#            for j in range(vert_num):
#                x_[j,:] = j*np.ones([1, num_samples])
#            
#            idx_x = tf.Variable(x_, trainable=False, dtype=tf.int32)
#            adj_ids = tf.nn.embedding_lookup(self.adj_info, ids)
#            adj_lists = tf.gather_nd(adj_ids, tf.stack([tf.reshape(idx_x,[-1]), tf.reshape(idx_y,[-1])], axis=1))
#            
#            adj_lists = tf.reshape(adj_lists, [vert_num, num_samples])
#
#            self.reuse = True


        #return adj_lists

        #return adj_lists, att_lists 




