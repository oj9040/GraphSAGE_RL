from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np

from graphsage.models import SampleAndAggregate, SAGEInfo, Node2VecModel
from graphsage.minibatch import EdgeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler, MLNeighborSampler, FastMLNeighborSampler
from graphsage.utils import load_data

from scipy import sparse
import pdb

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage', 'model names. See README for possible values.')  
flags.DEFINE_float('learning_rate', 0.00001, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'name of the object file that stores the training data. must be specified.')
flags.DEFINE_string('model_prefix', '', 'model idx.')

# sampler param
flags.DEFINE_boolean('nonlinear_sampler', False, 'Where to use nonlinear sampler o.w. linear sampler')
flags.DEFINE_float('uniform_ratio', 0.6, 'In case of FastML sampling, the percentile of uniform sampling preceding the regressor sampling')

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 1, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of users samples in layer 2')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('neg_sample_size', 20, 'number of negative samples')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_integer('n2v_test_epochs', 1, 'Number of new SGD epochs for n2v.')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', False, 'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 50, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def log_dir(sampler_model_name):
    log_dir = FLAGS.base_log_dir + "/output/unsup-" + FLAGS.train_prefix.split("/")[-2] + '-' + FLAGS.model_prefix + '-' + sampler_model_name
    log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def sampler_log_dir():
    log_dir = FLAGS.base_log_dir + "/output/sampler-unsup-" + FLAGS.train_prefix.split("/")[-2] + '-' + FLAGS.model_prefix
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
 
# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val = minibatch_iter.val_feed_dict(size)
    outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

def incremental_evaluate(sess, model, minibatch_iter, size):
    t_test = time.time()
    finished = False
    val_losses = []
    val_mrrs = []
    iter_num = 0
    while not finished:
        feed_dict_val, finished, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                            feed_dict=feed_dict_val)
        val_losses.append(outs_val[0])
        val_mrrs.append(outs_val[2])
    return np.mean(val_losses), np.mean(val_mrrs), (time.time() - t_test)

def save_val_embeddings(sess, model, minibatch_iter, size, out_dir, mod=""):
    val_embeddings = []
    finished = False
    seen = set([])
    nodes = []
    iter_num = 0
    name = "val"
    while not finished:
        pdb.set_trace()
        feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.mrr, model.outputs1], 
                            feed_dict=feed_dict_val)
        #ONLY SAVE FOR embeds1 because of planetoid
        for i, edge in enumerate(edges):
            if not edge[0] in seen:
                val_embeddings.append(outs_val[-1][i,:])
                nodes.append(edge[0])
                seen.add(edge[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    val_embeddings = np.vstack(val_embeddings)
    np.save(out_dir + name + mod + ".npy",  val_embeddings)
    with open(out_dir + name + mod + ".txt", "w") as fp:
        fp.write("\n".join(map(str,nodes)))

def construct_placeholders():
    # Define placeholders
    placeholders = {
        'batch1' : tf.placeholder(tf.int32, shape=(FLAGS.batch_size), name='batch1'),
        'batch2' : tf.placeholder(tf.int32, shape=(FLAGS.batch_size), name='batch2'),
 
       # 'batch1' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
       # 'batch2' : tf.placeholder(tf.int32, shape=(None), name='batch2'),
        # negative samples for all nodes in the batch
        'neg_samples': tf.placeholder(tf.int32, shape=(None,),
            name='neg_sample_size'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

def train(train_data, test_data=None, sampler_name='Uniform'):
    
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders()
    minibatch = EdgeMinibatchIterator(G, 
            id_map,
            placeholders, batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree, 
            num_neg_samples=FLAGS.neg_sample_size,
            context_pairs = context_pairs)
    
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
    adj_shape = adj_info.get_shape().as_list()

    if FLAGS.model == 'mean_concat':
        # Create model

        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)


        #sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     concat=True,
                                     layer_infos=layer_infos, 
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    
    elif FLAGS.model == 'mean_add':
        # Create model

        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)


        #sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     concat=False,
                                     layer_infos=layer_infos, 
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'gcn':

        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)

           
        # Create model
        #sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="gcn",
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     concat=False,
                                     logging=True)

    elif FLAGS.model == 'graphsage_seq':
        
        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)


        #sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     identity_dim = FLAGS.identity_dim,
                                     aggregator_type="seq",
                                     model_size=FLAGS.model_size,
                                     logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        
        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)
    
            
        #sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="maxpool",
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    elif FLAGS.model == 'graphsage_meanpool':
        
        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)
            
        #sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="meanpool",
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'n2v':
        model = Node2VecModel(placeholders, features.shape[0],
                                       minibatch.deg,
                                       #2x because graphsage uses concat
                                       nodevec_dim=2*FLAGS.dim_1,
                                       lr=FLAGS.learning_rate)
    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    
    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(sampler_name), sess.graph)
    
    
    # Save model
    saver = tf.train.Saver()
    model_path =  './model/unsup-' + FLAGS.train_prefix.split('/')[-1] + '-' + FLAGS.model_prefix + '-' + sampler_name
    model_path += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    
    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
    
    
    # Restore params of ML sampler model
    if sampler_name == 'ML' or sampler_name == 'FastML':
        sampler_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="MLsampler")
        #pdb.set_trace() 
        saver_sampler = tf.train.Saver(var_list=sampler_vars)
        sampler_model_path = './model/MLsampler-unsup-' + FLAGS.train_prefix.split('/')[-1] + '-' + FLAGS.model_prefix
        sampler_model_path += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)

        saver_sampler.restore(sess, sampler_model_path + 'model.ckpt')

   
    # Train model
    
    train_shadow_mrr = None
    shadow_mrr = None

    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)

    val_cost_ = []
    val_mrr_ = []
    shadow_mrr_ = []
    duration_ = []
    
    ln_acc = sparse.csr_matrix((adj_shape[0], adj_shape[0]), dtype=np.float32)
    lnc_acc = sparse.csr_matrix((adj_shape[0], adj_shape[0]), dtype=np.int32)
    
    ln_acc = ln_acc.tolil()
    lnc_acc = lnc_acc.tolil()

    for epoch in range(FLAGS.epochs): 
        minibatch.shuffle() 

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict = minibatch.next_minibatch_feed_dict()

            if feed_dict.values()[0] != FLAGS.batch_size:
                break

            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all, 
                    model.mrr, model.outputs1, model.loss_node, model.loss_node_count], feed_dict=feed_dict)
            train_cost = outs[2]
            train_mrr = outs[5]
            
            if train_shadow_mrr is None:
                train_shadow_mrr = train_mrr#
            else:
                train_shadow_mrr -= (1-0.99) * (train_shadow_mrr - train_mrr)

            if iter % FLAGS.validate_iter == 0:
                # Validation
                sess.run(val_adj_info.op)
                val_cost, ranks, val_mrr, duration  = evaluate(sess, model, minibatch, size=FLAGS.validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost
            
            if shadow_mrr is None:
                shadow_mrr = val_mrr
            else:
                shadow_mrr -= (1-0.99) * (shadow_mrr - val_mrr)
            
            val_cost_.append(val_cost)
            val_mrr_.append(val_mrr)
            shadow_mrr_.append(shadow_mrr)
            duration_.append(duration)


            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)
    
            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                print("Iter:", '%04d' % iter, 
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_mrr=", "{:.5f}".format(train_mrr), 
                      "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr), # exponential moving average
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_mrr=", "{:.5f}".format(val_mrr), 
                      "val_mrr_ema=", "{:.5f}".format(shadow_mrr), # exponential moving average
                      "time=", "{:.5f}".format(avg_time))

            
            ln = outs[7].values
            ln_idx = outs[7].indices
            ln_acc[ln_idx[:,0], ln_idx[:,1]] += ln

            lnc = outs[8].values
            lnc_idx = outs[8].indices
            lnc_acc[lnc_idx[:,0], lnc_idx[:,1]] += lnc
                
                
            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
                break


    print("Validation per epoch in training")
    for ep in range(FLAGS.epochs):
        print("Epoch: %04d"%ep, " val_cost={:.5f}".format(val_cost_[ep]), " val_mrr={:.5f}".format(val_mrr_[ep]), " val_mrr_ema={:.5f}".format(shadow_mrr_[ep]), " duration={:.5f}".format(duration_[ep]))
 
    print("Optimization Finished!")
    
    # Save model
    save_path = saver.save(sess, model_path+'model.ckpt')
    print ('model is saved at %s'%save_path)


    # Save loss node and count
    loss_node_path = './loss_node/unsup-' + FLAGS.train_prefix.split('/')[-1] + '-' + FLAGS.model_prefix + '-' + sampler_name
    loss_node_path += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(loss_node_path):
        os.makedirs(loss_node_path)

    loss_node = sparse.save_npz(loss_node_path + 'loss_node.npz', sparse.csr_matrix(ln_acc))
    loss_node_count = sparse.save_npz(loss_node_path + 'loss_node_count.npz', sparse.csr_matrix(lnc_acc))
    print ('loss and count per node is saved at %s'%loss_node_path)    
    
    
    
    if FLAGS.save_embeddings:
        sess.run(val_adj_info.op)

        save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir(sampler_name))

        if FLAGS.model == "n2v":
            # stopping the gradient for the already trained nodes
            train_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if not G.node[n]['val'] and not G.node[n]['test']],
                    dtype=tf.int32)
            test_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if G.node[n]['val'] or G.node[n]['test']], 
                    dtype=tf.int32)
            update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(test_ids))
            no_update_nodes = tf.nn.embedding_lookup(model.context_embeds,tf.squeeze(train_ids))
            update_nodes = tf.scatter_nd(test_ids, update_nodes, tf.shape(model.context_embeds))
            no_update_nodes = tf.stop_gradient(tf.scatter_nd(train_ids, no_update_nodes, tf.shape(model.context_embeds)))
            model.context_embeds = update_nodes + no_update_nodes
            sess.run(model.context_embeds)

            # run random walks
            from graphsage.utils import run_random_walks
            nodes = [n for n in G.nodes_iter() if G.node[n]["val"] or G.node[n]["test"]]
            start_time = time.time()
            pairs = run_random_walks(G, nodes, num_walks=50)
            walk_time = time.time() - start_time

            test_minibatch = EdgeMinibatchIterator(G, 
                id_map,
                placeholders, batch_size=FLAGS.batch_size,
                max_degree=FLAGS.max_degree, 
                num_neg_samples=FLAGS.neg_sample_size,
                context_pairs = pairs,
                n2v_retrain=True,
                fixed_n2v=True)
            
            start_time = time.time()
            print("Doing test training for n2v.")
            test_steps = 0
            for epoch in range(FLAGS.n2v_test_epochs):
                test_minibatch.shuffle()
                while not test_minibatch.end():
                    feed_dict = test_minibatch.next_minibatch_feed_dict()
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    outs = sess.run([model.opt_op, model.loss, model.ranks, model.aff_all, 
                        model.mrr, model.outputs1], feed_dict=feed_dict)
                    if test_steps % FLAGS.print_every == 0:
                        print("Iter:", '%04d' % test_steps, 
                              "train_loss=", "{:.5f}".format(outs[1]),
                              "train_mrr=", "{:.5f}".format(outs[-2]))
                    test_steps += 1
            train_time = time.time() - start_time
            save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir(sampler_name), mod="-test")
            print("Total time: ", train_time+walk_time)
            print("Walk time: ", walk_time)
            print("Train time: ", train_time)


# Sampler
def train_sampler(train_data):

    features = train_data[1]
    #batch_size = FLAGS.batch_size
    batch_size = 512

    if not features is None:
        features = np.vstack([features, np.zeros((features.shape[1],))])
   
    node_size = len(features)
    node_dim = len(features[0])

    # build model
    # input (features of vertex and its neighbor, label)
    x1_ph = tf.placeholder(shape=[batch_size, node_dim], dtype=tf.float32)
    x2_ph = tf.placeholder(shape=[batch_size, node_dim], dtype=tf.float32) 
    y_ph = tf.placeholder(shape=[batch_size], dtype=tf.float32)
    
    with tf.variable_scope("MLsampler"):
        
        if FLAGS.nonlinear_sampler == True:
       
            print ("Non-linear regression sampler used")

            l = tf.layers.dense(tf.concat([x1_ph, x2_ph], axis=1), 1, activation=None, trainable=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense')
      
            out = tf.nn.relu(tf.exp(l), name='relu')
        else:

            print ("Linear regression sampler used")
           
            l = tf.layers.dense(x1_ph, node_dim, activation=None, trainable=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense')
            
            l = tf.matmul(l, x2_ph, transpose_b=True, name='matmul')
            out = tf.nn.relu(l, name='relu')
        ###

    loss = tf.nn.l2_loss(out-y_ph, name='loss')/batch_size
    
    '''
    with tf.variable_scope("MLsampler"):
        #bias = tf.Variable(tf.zeros([1]), trainable=True, name='bias')
        # layer 
        # relu(x1*W*x2)
       
        #drop_rate = 0.5
        #x1_ph = tf.layers.dropout(x1_ph, rate=drop_rate, training=True) 
        l = tf.layers.dense(x1_ph, node_dim, activation=None, trainable=True, name='dense')
        #l = tf.nn.relu(l, name='relu')

        #l = tf.layers.dense(l, node_dim, activation=None, trainable=True, name='dense2')
        #l = tf.nn.relu(l, name='relu')

        l = tf.matmul(l, x2_ph, transpose_b=True, name='matmul')
        
        #l = tf.nn.bias_add(l, tf.tile(bias, [batch_size]))
        out = tf.nn.relu(l, name='relu')
        ###

    loss = tf.nn.l2_loss(out-y_ph, name='loss')/batch_size
    '''
     
#    sampler_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="MLsampler")
#    for var in sampler_vars:
#        loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
#
    
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, name='Adam').minimize(loss)
    init = tf.global_variables_initializer()


    # configuration
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

  
    # load data
    loss_node_path = './loss_node/unsup-' + FLAGS.train_prefix.split('/')[-1] + '-' + FLAGS.model_prefix + '-Uniform'
    loss_node_path += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)

    loss_node = sparse.load_npz(loss_node_path + 'loss_node.npz')
    loss_node_count = sparse.load_npz(loss_node_path + 'loss_node_count.npz')

    #idx_nz = np.where(loss_node_count != 0)
    
    #pdb.set_trace()
    
    idx_nz = sparse.find(loss_node_count)
   
    # due to out of memory, select randomly limited number of data node

    if FLAGS.train_prefix.split('/')[-1] == 'reddit':

        perm = np.random.permutation(idx_nz[0].shape[0])
        perm = perm[:200000]
        vertex = features[idx_nz[0][perm]]
        neighbor = features[idx_nz[1][perm]]
        count = idx_nz[2][perm]
        y = np.divide(sparse.find(loss_node)[2][perm],count)

    else:

        vertex = features[idx_nz[0]]
        neighbor = features[idx_nz[1]]
        count = idx_nz[2]
        y = np.divide(sparse.find(loss_node)[2],count)

   
    # partition train/validation data
    vertex_tr = vertex[:-batch_size]
    neighbor_tr = neighbor[:-batch_size]
    y_tr = y[:-batch_size]

    vertex_val = vertex[-batch_size:] 
    neighbor_val = neighbor[-batch_size:]
    y_val = y[-batch_size:]

    iter_size = int(vertex_tr.shape[0]/batch_size)

    # initialize session
    sess = tf.Session(config=config)
    # summary
    tf.summary.scalar('loss', loss)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(sampler_log_dir(), sess.graph)


    # save model
    saver = tf.train.Saver()
    model_path = './model/MLsampler-unsup-' + FLAGS.train_prefix.split('/')[-1] + '-' + FLAGS.model_prefix
    model_path += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # init variables
    sess.run(init)
        
    # train
    total_steps = 0
    avg_time = 0.0
    
    #for epoch in range(50):
    for epoch in range(FLAGS.epochs):
        
        # shuffle
        perm = np.random.permutation(vertex_tr.shape[0])

        print("Epoch: %04d" %(epoch+1))

        for iter in range(iter_size):
                    
            # allocate batch
            vtr = vertex_tr[perm[iter*batch_size:(iter+1)*batch_size]]
            ntr = neighbor_tr[perm[iter*batch_size:(iter+1)*batch_size]]
            ytr = y_tr[perm[iter*batch_size:(iter+1)*batch_size]]

            t = time.time()
            outs = sess.run([ loss, optimizer, merged_summary_op], feed_dict={x1_ph: vtr, x2_ph: ntr, y_ph: ytr})
            train_loss = outs[0]
           

            # validation
            if iter%FLAGS.validate_iter == 0:

                outs = sess.run([ loss, optimizer, merged_summary_op], feed_dict={x1_ph: vertex_val, x2_ph: neighbor_val, y_ph: y_val})  
                val_loss = outs[0]

                
            avg_time = (avg_time*total_steps+time.time() - t)/(total_steps+1)

            # print 
            if total_steps%FLAGS.print_every == 0:
                print("Iter:", "%04d"%iter,
                                "train_loss=", "{:.5f}".format(train_loss),
                                "val_loss=", "{:.5f}".format(val_loss))

            total_steps +=1
            
            if total_steps > FLAGS.max_total_steps:
                break
        
        # save_model
        save_path = saver.save(sess, model_path+'model.ckpt')
        print ('model is saved at %s'%save_path)

    sess.close()
    tf.reset_default_graph()

   

def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix, load_walks=True)
    print("Done loading training data..")

    print("Start training uniform sampling + graphsage model..")
    train(train_data, sampler_name='Uniform')
    print("Done training uniform sampling + graphsage model..")

    print("Start training ML sampler..")
    train_sampler(train_data)
    print("Done training ML sampler..")

    print("Start training ML sampling + graphsage model..")
    train(train_data, sampler_name='FastML')
    print("Done training ML sampling + graphsage model..")


if __name__ == '__main__':
    tf.app.run()
