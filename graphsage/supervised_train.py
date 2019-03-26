from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics

from graphsage.supervised_models import SupervisedGraphsage
from graphsage.models import SAGEInfo
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler, MLNeighborSampler, FastMLNeighborSampler
from graphsage.utils import load_data

from scipy import sparse 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
from tensorflow.python.client import timeline

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
flags.DEFINE_string('model', 'mean_add', 'model names. See README for possible values.')  
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')
flags.DEFINE_string('model_prefix', '', 'model idx.')
flags.DEFINE_string('hyper_prefix', '', 'hyper idx.')


# sampler param
flags.DEFINE_boolean('nonlinear_sampler', True, 'Whether to use nonlinear sampler o.w. linear sampler')
flags.DEFINE_boolean('fast_ver', False, 'Whether to use a fast version of nonlinear_sampler')
flags.DEFINE_boolean('allhop_rewards', False, 'Whether to use a all-hop rewards or last-hop reward for training nonlinear_sampler')


# left to default values in main experiments 
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_3', 0, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 512, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")
flags.DEFINE_boolean('timeline', False, 'export timeline')

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def model_prefix():
    model_prefix = 'f' + str(FLAGS.dim_1) + '_' + str(FLAGS.dim_2) + '_' + str(FLAGS.dim_3) + '-s' + str(FLAGS.samples_1) + '_' + str(FLAGS.samples_2) + '_' + str(FLAGS.samples_3) 
    return model_prefix

def hyper_prefix():
    hyper_prefix = "/{model:s}-{model_size:s}-lr{lr:0.4f}-bs{batch_size:d}-ep{epochs:d}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate,
            batch_size=FLAGS.batch_size,
            epochs=FLAGS.epochs)
    return hyper_prefix

# calculate only micro f1 score
def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
   
    f1_micro = metrics.f1_score(y_true, y_pred, average="micro")
    return f1_micro, f1_micro 

# calculate micro and macro f1 score
def calc_f1__(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    #pdb.set_trace()
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss], 
                        feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    
    return node_outs_val[1], mic, mac, (time.time() - t_test)

def log_dir(sampler_model_name):
    log_dir = FLAGS.base_log_dir + "/output/sup-" + FLAGS.train_prefix.split("/")[-2] + '-' + model_prefix() + '-' + str(sampler_model_name)

    log_dir += hyper_prefix()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def sampler_log_dir():
    log_dir = FLAGS.base_log_dir + "/output/sampler-sup-" + FLAGS.train_prefix.split("/")[-2] + '-' + model_prefix()
   
    log_dir += hyper_prefix()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def incremental_evaluate(sess, model, minibatch_iter, size, run_options=None, run_metadata=None, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    
    while not finished:
        feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        
        if feed_dict_val.values()[0] != FLAGS.batch_size:
            break

        node_outs_val = sess.run([model.preds, model.loss], 
                         feed_dict=feed_dict_val, options=run_options, run_metadata=run_metadata)

        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)

    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)


def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch' : tf.placeholder(tf.int32, shape=(FLAGS.batch_size), name='batch1'),
        #'batch' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
        'learning_rate': tf.placeholder(tf.float32, name='learning_rate')
    }
    return placeholders

def train(train_data, test_data=None, sampler_name='Uniform'):
    
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map  = train_data[4]

    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders(num_classes)
    minibatch = NodeMinibatchIterator(G, 
            id_map,
            placeholders, 
            class_map,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree, 
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

        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_3)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]
       
        # modified
        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos,
                                     concat=True,
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
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

        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_3)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]
       
        # modified
        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos,
                                     concat=False,
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'gcn':
        
        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features)

            
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]
        
        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="gcn",
                                     model_size=FLAGS.model_size,
                                     concat=False,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="seq",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
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

        model = SupervisedGraphsage(num_classes, placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="maxpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
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

        model = SupervisedGraphsage(num_classes, placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="meanpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

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
    model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(var_list=model_vars)

    model_path = './model/' + FLAGS.train_prefix.split("/")[-1] + '-' + model_prefix() + '-' + sampler_name
    model_path += hyper_prefix()
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
   

    # Restore params of ML sampler model
    if sampler_name == 'ML' or sampler_name == 'FastML':
        sampler_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="MLsampler")
        #pdb.set_trace() 
        saver_sampler = tf.train.Saver(var_list=sampler_vars)
        
        if FLAGS.allhop_rewards:
            sampler_model_path = './model/MLsampler-' + FLAGS.train_prefix.split('/')[-1] + '-' + model_prefix() + '-allhops'
        else:
            sampler_model_path = './model/MLsampler-' + FLAGS.train_prefix.split('/')[-1] + '-' + model_prefix() + '-lasthop'
        
        sampler_model_path += hyper_prefix()
        
        saver_sampler.restore(sess, sampler_model_path + 'model.ckpt')

    # Train model
    
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    
    
    val_cost_ = []
    val_f1_mic_ = []
    val_f1_mac_ = []
    duration_ = []
    epoch_laps_ = []

    ln_acc = sparse.csr_matrix((adj_shape[0], adj_shape[0]), dtype=np.float32)
    lnc_acc = sparse.csr_matrix((adj_shape[0], adj_shape[0]), dtype=np.int32)
    
    ln_acc = ln_acc.tolil()
    lnc_acc = lnc_acc.tolil()

    #learning_rate = [0.01, 0.001, 0.0001]
    learning_rate = [FLAGS.learning_rate]
    
    for lr_iter in range(len(learning_rate)):

            for epoch in range(FLAGS.epochs): 
                
                epoch_time = time.time()
                    
                minibatch.shuffle() 

                iter = 0
                print('Epoch: %04d' % (epoch + 1))
                epoch_val_costs.append(0)
                
                while not minibatch.end():
                    # Construct feed dictionary
                    feed_dict, labels = minibatch.next_minibatch_feed_dict()
                    

                    if feed_dict.values()[0] != FLAGS.batch_size:
                        break
                    

                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    feed_dict.update({placeholders['learning_rate']: learning_rate[lr_iter]})

                    t = time.time()
                    
                    # Training step
                    outs = sess.run([merged, model.opt_op, model.loss, model.preds, model.loss_node, model.loss_node_count], feed_dict=feed_dict)
                    train_cost = outs[2]


                    if iter % FLAGS.validate_iter == 0:
                        # Validation
                        sess.run(val_adj_info.op)
                        if FLAGS.validate_batch_size == -1:
                            val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
                        else:
                            val_cost, val_f1_mic, val_f1_mac, duration = evaluate(sess, model, minibatch, FLAGS.validate_batch_size)
                        
                        # accumulate val results
                        val_cost_.append(val_cost)
                        val_f1_mic_.append(val_f1_mic)
                        val_f1_mac_.append(val_f1_mac)
                        duration_.append(duration)

                        #
                        sess.run(train_adj_info.op)
                        epoch_val_costs[-1] += val_cost


                    if total_steps % FLAGS.print_every == 0:
                        summary_writer.add_summary(outs[0], total_steps)
            
                    # Print results
                    avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

                    
                    ln = outs[4].values
                    ln_idx = outs[4].indices
                    ln_acc[ln_idx[:,0], ln_idx[:,1]] += ln
                   

                    lnc = outs[5].values
                    lnc_idx = outs[5].indices
                    lnc_acc[lnc_idx[:,0], lnc_idx[:,1]] += lnc
                   
                    if total_steps % FLAGS.print_every == 0:
                        train_f1_mic, train_f1_mac = calc_f1(labels, outs[3])
                        print("Iter:", '%04d' % iter, 
                              "train_loss=", "{:.5f}".format(train_cost),
                              "train_f1_mic=", "{:.5f}".format(train_f1_mic), 
                              "val_loss=", "{:.5f}".format(val_cost),
                              "val_f1_mic=", "{:.5f}".format(val_f1_mic), 
                              "time per iter=", "{:.5f}".format(avg_time))
                       
                    iter += 1
                    total_steps += 1

                    if total_steps > FLAGS.max_total_steps:
                        break

                epoch_laps = time.time()-epoch_time
                epoch_laps_.append(epoch_laps)
                print("Epoch time=", "{:.5f}".format(epoch_laps))

                if total_steps > FLAGS.max_total_steps:
                        break
    
    print("avg time per epoch=", "{:.5f}".format(np.mean(epoch_laps_)))   


    # Save model
    save_path = saver.save(sess, model_path+'model.ckpt')
    print ('model is saved at %s'%save_path)


    # Save loss node and count
    loss_node_path = './loss_node/' + FLAGS.train_prefix.split('/')[-1] + '-' + model_prefix() + '-' + sampler_name
    loss_node_path += hyper_prefix()
    
    if not os.path.exists(loss_node_path):
        os.makedirs(loss_node_path)

    loss_node = sparse.save_npz(loss_node_path + 'loss_node.npz', sparse.csr_matrix(ln_acc))
    loss_node_count = sparse.save_npz(loss_node_path + 'loss_node_count.npz', sparse.csr_matrix(lnc_acc))
    print ('loss and count per node is saved at %s'%loss_node_path)    
 

    print("Optimization Finished!")
    sess.run(val_adj_info.op)
    
    
    # test 
    val_cost_ = []
    val_f1_mic_ = []
    val_f1_mac_ = []
    duration_ = []

    print("Writing test set stats to file (don't peak!)")
    
    # timeline
    if FLAGS.timeline == True:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    else:
        run_options = None
        run_metadata = None

    for iter in range(10):
        
        val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size, run_options, run_metadata, test=True)
        
        print("Full validation stats:",
                          "loss=", "{:.5f}".format(val_cost),
                          "f1_micro=", "{:.5f}".format(val_f1_mic),
                          "time=", "{:.5f}".format(duration))

        val_cost_.append(val_cost)
        val_f1_mic_.append(val_f1_mic)
        duration_.append(duration)



    print("mean: loss={:.5f} f1_micro={:.5f} time={:.5f}\n".format(np.mean(val_cost_), np.mean(val_f1_mic_), np.mean(duration_)))
    
    # write test results
    with open(log_dir(sampler_name) + "test_stats.txt", "w") as fp:
        for iter in range(10):
            fp.write("loss={:.5f} f1_micro={:.5f} time={:.5f}\n".
                        format(val_cost_[iter], val_f1_mic_[iter], duration_[iter]))
        
        fp.write("mean: loss={:.5f} f1_micro={:.5f} time={:.5f}\n".
                        format(np.mean(val_cost_), np.mean(val_f1_mic_), np.mean(duration_)))
        fp.write("variance: loss={:.5f} f1_micro={:.5f} time={:.5f}\n".
                        format(np.var(val_cost_), np.var(val_f1_mic_), np.var(duration_)))

    
    # create timeline object, and write it to a json
    if FLAGS.timeline == True:
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format(show_memory=True)
        with open(log_dir(sampler_name) + 'timeline.json', 'w') as f:
            print ('timeline written at %s'%(log_dir(sampler_name)+'timelnie.json'))
            f.write(ctf)

  
    sess.close()
    tf.reset_default_graph()

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
    lr_ph = tf.placeholder(dtype=tf.float32)


    # sampler model (non-linear, linear)
    with tf.variable_scope("MLsampler"):
        
        if FLAGS.nonlinear_sampler == True:
       
            print ("Non-linear regression sampler used")

            l = tf.layers.dense(tf.concat([x1_ph, x2_ph], axis=1), 1, activation=tf.nn.relu, trainable=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense')
      
            out = tf.exp(l)

        else:

            print ("Linear regression sampler used")
           
            l = tf.layers.dense(x1_ph, node_dim, activation=None, trainable=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense')
            
            l = tf.matmul(l, x2_ph, transpose_b=True, name='matmul')
            out = tf.nn.relu(l, name='relu')
        ###

    # l2 loss
    loss = tf.nn.l2_loss(out-y_ph, name='loss')/batch_size

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_ph, name='Adam').minimize(loss)

    # initialize global variables
    init = tf.global_variables_initializer()

    # configuration
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    # construct reward from loss
    if FLAGS.allhop_rewards:
        # using all-hop rewards
        
        dims = [FLAGS.dim_1, FLAGS.dim_2, FLAGS.dim_3]    
        samples = [FLAGS.samples_1, FLAGS.samples_2, FLAGS.samples_3]

        numhop = np.count_nonzero(samples)
        gamma = 0.8
        loss_node = 0
        loss_node_count = 0
        for i in reversed(range(0, numhop)):
           
            model_prefix_ = 'f' + str(dims[0]) + '_' + str(dims[1]) + '_' + str(dims[2]) + '-s' + str(samples[0]) + '_' + str(samples[1]) + '_' + str(samples[2]) 

            # load data
            loss_node_path = './loss_node/' + FLAGS.train_prefix.split('/')[-1] + '-' + model_prefix_ + '-Uniform'
            loss_node_path += hyper_prefix()
            
            loss_node_perstep = sparse.load_npz(loss_node_path + 'loss_node.npz')
            loss_node_count_perstep = sparse.load_npz(loss_node_path + 'loss_node_count.npz')


            loss_node += (gamma**i)*loss_node_perstep
            loss_node_count += loss_node_count_perstep
            
            dims[i] = 0
            samples[i] = 0

    else:
        # using only last-hop reward
        
        # load data
        loss_node_path = './loss_node/' + FLAGS.train_prefix.split('/')[-1] + '-' + model_prefix() + '-Uniform'
        loss_node_path += hyper_prefix()

        loss_node = sparse.load_npz(loss_node_path + 'loss_node.npz')
        loss_node_count = sparse.load_npz(loss_node_path + 'loss_node_count.npz')

   
    cnt_nz = sparse.find(loss_node_count)
    loss_nz = sparse.find(loss_node)


    # subsampling if the number of loss nodes is very large
    if cnt_nz[0].shape[0] > 1000000:
        cnt_nz_samp = np.int32(np.random.uniform(0, cnt_nz[0].shape[0]-1, 1000000))
        cnt_nz_v = cnt_nz[0][cnt_nz_samp]
        cnt_nz_n = cnt_nz[1][cnt_nz_samp]
        cnt = cnt_nz[2][cnt_nz_samp]
        lss = loss_nz[2][cnt_nz_samp]
        
    else:
        cnt_nz_v = cnt_nz[0]
        cnt_nz_n = cnt_nz[1]
        cnt = cnt_nz[2]
        lss = loss_nz[2]

    vertex = features[cnt_nz_v]
    neighbor = features[cnt_nz_n]
    y = np.divide(lss, cnt)
    
    # plot histogram of reward
    fig = plt.hist(y, bins=128, range=(0,np.mean(y)*2), alpha=0.7, color='k')
    plt.xlabel('Value')
    plt.ylabel('Number')
    plt.savefig(loss_node_path + 'histogram_valuefunc.png')
    
    #plt.clf()
    #y = (y-np.min(y))/(np.mean(y)-np.min(y))*255
    #fig = plt.hist(y, bins=128, range=(0,1000), alpha=0.7, color='k')
    #plt.xlabel('Value')
    #plt.ylabel('Number')
    #plt.savefig(loss_node_path + 'histogram_valuefunc_norm.png')

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
    model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(var_list=model_vars)
    #saver = tf.train.Saver()
    if FLAGS.allhop_rewards:
        model_path = './model/MLsampler-' + FLAGS.train_prefix.split('/')[-1] + '-' + model_prefix() + '-allhops'
    else:
        model_path = './model/MLsampler-' + FLAGS.train_prefix.split('/')[-1] + '-' + model_prefix() + '-lasthop'
    model_path += hyper_prefix()
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # init variables
    sess.run(init)
        
    # train
    total_steps = 0
    avg_time = 0.0
     
    # learning rate of sampler needs to be smaller than gnn's for stable optimization
    lr = [0.001]

    val_loss_old = 0

    for lr_iter in range(len(lr)):

        print ('learning rate= %f'%lr[lr_iter])
        #optimizer = tf.train.AdamOptimizer(learning_rate=lr[lr_iter], name='Adam').minimize(loss)
            
        #for epoch in range(epochs):
        for epoch in range(50):

            # shuffle
            perm = np.random.permutation(vertex_tr.shape[0])

            print("Epoch: %04d" %(epoch+1))

            for iter in range(iter_size):
                            
                # allocate batch
                vtr = vertex_tr[perm[iter*batch_size:(iter+1)*batch_size]]
                ntr = neighbor_tr[perm[iter*batch_size:(iter+1)*batch_size]]
                ytr = y_tr[perm[iter*batch_size:(iter+1)*batch_size]]

                t = time.time()
                outs = sess.run([ loss, optimizer, merged_summary_op], feed_dict={x1_ph: vtr, x2_ph: ntr, y_ph: ytr, lr_ph: lr[lr_iter]})
                train_loss = outs[0]
                   

                # validation
                if iter%FLAGS.validate_iter == 0:

                    outs = sess.run([ loss, optimizer, merged_summary_op], feed_dict={x1_ph: vertex_val, x2_ph: neighbor_val, y_ph: y_val, lr_ph: lr[lr_iter]})  
                    val_loss = outs[0]
                    
                    if val_loss == val_loss_old:
                        
                        sess.close()
                        tf.reset_default_graph()
                        return 0
                    else:
                        val_loss_old = val_loss
                        
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

    ## train graphsage model
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix)
    print("Done loading training data..")

    print("Start training uniform sampling + graphsage model..")
    if FLAGS.allhop_rewards:
        
        dim_2_org = FLAGS.dim_2
        dim_3_org = FLAGS.dim_3
        samples_2_org = FLAGS.samples_2
        samples_3_org = FLAGS.samples_3
            
        dims = [FLAGS.dim_1, FLAGS.dim_2, FLAGS.dim_3]
        samples = [FLAGS.samples_1, FLAGS.samples_2, FLAGS.samples_3]
        numhop = np.count_nonzero(samples)
        for i in reversed(range(0, numhop)):
            
            FLAGS.dim_2 = dims[1]
            FLAGS.dim_3 = dims[2]
            FLAGS.samples_2 = samples[1]
            FLAGS.samples_3 = samples[2]
            print ('Obtainining %d/%d hop reward'%(i+1, numhop)) 
            train(train_data, sampler_name='Uniform')
            
            dims[i] = 0
            samples[i] = 0
        
        FLAGS.dim_2 = dim_2_org
        FLAGS.dim_3 = dim_3_org
        FLAGS.samples_2 = samples_2_org
        FLAGS.samples_3 = samples_3_org

    else:
        train(train_data, sampler_name='Uniform')
    print("Done training uniform sampling + graphsage model..")

    ## train sampler
    print("Start training ML sampler..")
    train_sampler(train_data)
    print("Done training ML sampler..")

    ## train 
    print("Start training ML sampling + graphsage model..")
    if FLAGS.fast_ver: 
        train(train_data, sampler_name='FastML')
    else:
        train(train_data, sampler_name='ML')
    print("Done training ML sampling + graphsage model..")

if __name__ == '__main__':
    tf.app.run()
