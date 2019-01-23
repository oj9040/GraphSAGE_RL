import tensorflow as tf

import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator, LRMeanAggregator, LogicMeanAggregator, AttMeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            placeholders, features, adj, degrees, layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "LRmean":
            self.aggregator_cls = LRMeanAggregator
        elif aggregator_type == "logicmean":
            self.aggregator_cls = LogicMeanAggregator
        elif aggregator_type == "attmean":
            self.aggregator_cls = AttMeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        
        # added
        #self.loss_node = loss_node
        #self.loss_node_count = loss_node_count
        #self.loss_node = tf.Variable(tf.zeros([adj.shape[0], adj.shape[0]]), trainable=False, name="loss_node", dtype=tf.float32)
        #self.loss_node_count = tf.Variable(tf.zeros([adj.shape[0], adj.shape[0]]), trainable=False, name="loss_node_count", dtype=tf.float32) 

        adj_shape = adj.get_shape().as_list()
        self.loss_node = tf.SparseTensor(indices=np.empty((0,2), dtype=np.int64), values=[], dense_shape=[adj_shape[0], adj_shape[0]])
        self.loss_node_count = tf.SparseTensor(indices=np.empty((0,2), dtype=np.int64), values=[], dense_shape=[adj_shape[0], adj_shape[0]])
        

        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        #self.batch_size = placeholders["batch_size"]
        self.batch_size = FLAGS.batch_size
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()


    def build(self):

        samples1, support_sizes1, attentions1, num_nz, out_mean = self.sample(self.inputs1, self.layer_infos)
        self.att = attentions1
        self.out_mean = out_mean
        #samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples, support_sizes1, attentions1, num_nz, concat=self.concat, model_size=self.model_size)
        #self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples, support_sizes1, concat=self.concat, model_size=self.model_size)

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)

        dim_mult = 2 if self.concat else 1
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes, 
                dropout=self.placeholders['dropout'],
                act=lambda x : x)
        # TF graph management
        self.node_preds = self.node_pred(self.outputs1)

        self._loss()

        # added
        self.sparse_loss_to_node(samples1, support_sizes1, num_samples)

        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()
        #self.micro_f1 = self.calc_micro_f1()
#        
#    # added
#    def loss_to_adj(self, samples, support_size, num_samples):
#        import pdb
#        pdb.set_trace()
#        batch_size = self.batch_size
#
#        total_length = 0
#        for i in range(num_samples):
#            total_length += support_size[i+1]
#        total_length *= batch_size
#
#        link_vec = tf.Variable(tf.zeros([2,total_length]), trainable=False)
#
#        for i in range(support_size-1):
#            vert = samples[i]
#            neig = samples[i+1]
#            num_sample = support_size[i+1]/support_size[i]
#
#
#
#        link_vec = tf.Variable(tf.zeros(, trainable=False, 
#        for k in range(1, len(samples)):
#
#            vert = samples[k-1]
#            neig = samples[k]
#            link_vec = vert
#
#
# i           bv = batch_size*support_size[k-1]
#            bn = batch_size*support_size[k]
#
#            for l in range(batch_size):
#                vertex = tf.slice(samples, [0, bv*l], support_size[k-1])
#                neighbor = tf.slice(samples, [0, bn*l], support_size[k])
#                adj_v = tf.gather_nd(self.adj_info, [vertex])
#                idx_n = tf.where(tf.equal(adj_v, neighbor))  
#                tf.constant(self.loss, [support_size
#                tf.scatter_nd(idx_n, updates, shape)
#      
    # added
    def sparse_loss_to_node(self, samples, support_size, num_samples):
        
        batch_size = self.batch_size
        
        length = sum(support_size[1:])*batch_size
        node_dim = self.loss_node.get_shape().as_list()

        #discount = .9
        for k in range(1, 2):
        #for k in range(1, len(samples)):

            #import pdb
            #pdb.set_trace()

            x = tf.reshape(tf.tile(tf.expand_dims(samples[k-1], -1), [1, tf.cast(support_size[k]/support_size[k-1], tf.int32)]), [-1])
            x = tf.cast(x, tf.int64)
            y = samples[k]
            y = tf.cast(y, tf.int64)
            idx = tf.expand_dims(x*node_dim[0] + y,1)
        
            #loss = (discount**(k-1))*tf.reshape(tf.tile(tf.expand_dims(tf.reduce_sum(self.cross_entropy, 1), -1), [1, support_size[k]]), [-1])
            #import pdb
            #pdb.set_trace()
            if FLAGS.sigmoid == True:
                loss = tf.reshape(tf.tile(tf.expand_dims(tf.reduce_sum(self.cross_entropy, 1), -1), [1, support_size[k]]), [-1])
            else:
                loss = tf.reshape(tf.tile(tf.expand_dims(self.cross_entropy, -1), [1, support_size[k]]), [-1])
            scatter1 = tf.SparseTensor(idx, loss, tf.constant([node_dim[0]*node_dim[1]], dtype=tf.int64))
            scatter1 = tf.sparse_reshape(scatter1, tf.constant([node_dim[0], node_dim[1]]))
            self.loss_node = tf.sparse_add(self.loss_node, scatter1)


            ones = tf.reshape(tf.tile(tf.expand_dims(tf.ones(batch_size), -1), [1, support_size[k]]), [-1])
            scatter2 = tf.SparseTensor(idx, ones, tf.constant([node_dim[0]*node_dim[1]], dtype=tf.int64))
            scatter2 = tf.sparse_reshape(scatter2, tf.constant([node_dim[0], node_dim[1]]))
            self.loss_node_count = tf.sparse_add(self.loss_node_count, scatter2) 


#    def loss_to_node(self, samples, support_size, num_samples):
#        
#        batch_size = self.batch_size
#        
#        length = sum(support_size[1:])*batch_size
#        node_dim = self.loss_node.get_shape().as_list()
#
#        for k in range(1, len(samples)):
#
#            x = tf.reshape(tf.tile(tf.expand_dims(samples[k-1], -1), [1, support_size[k]/support_size[k-1]]), [-1])
#            x = tf.cast(x, tf.int64)
#            y = samples[k]
#            y = tf.cast(y, tf.int64)
#            idx = tf.expand_dims(x*node_dim[0] + y,1)
#            #idx = tf.stack([x,y], axis=1)
#          
#            loss = tf.reshape(tf.tile(tf.expand_dims(tf.reduce_sum(self.cross_entropy, 1), -1), [1, support_size[k]]), [-1])
#        
#
#            scatter = tf.scatter_nd(idx, loss, tf.constant([node_dim[0]*node_dim[1]], dtype=tf.int64))
#            scatter = tf.reshape(scatter, tf.constant([node_dim[0], node_dim[1]]))
#            self.loss_node = tf.assign_add(self.loss_node, scatter)
#
#            ones = tf.reshape(tf.tile(tf.expand_dims(tf.ones(batch_size), -1), [1, support_size[k]]), [-1])
#            scatter = tf.scatter_nd(idx, ones, tf.constant([node_dim[0]*node_dim[1]], dtype=tf.int64))
#            scatter = tf.reshape(scatter, tf.constant([node_dim[0], node_dim[1]]))
#            self.loss_node_count = tf.assign_add(self.loss_node_count, scatter) 
#            

    def _loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
       
        # classification loss
        if self.sigmoid_loss:
            
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.node_preds, labels=self.placeholders['labels'])
            self.loss += tf.reduce_mean(self.cross_entropy)
        else:
            
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels'])

            self.loss += tf.reduce_mean(self.cross_entropy)
       
        
        tf.summary.scalar('loss', self.loss)
        
        

        
        

#     def _loss(self):
#        # Weight decay loss
#        for aggregator in self.aggregators:
#            for var in aggregator.vars.values():
#                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
#        for var in self.node_pred.vars.values():
#            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
#       
#        # classification loss
#        if self.sigmoid_loss:
#            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#                    logits=self.node_preds,
#                    labels=self.placeholders['labels']))
#        else:
#            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#                    logits=self.node_preds,
#                    labels=self.placeholders['labels']))
#
#        tf.summary.scalar('loss', self.loss)
        



    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)


    '''
    def calc_micro_f1(self):
       

        import pdb
        pdb.set_trace()

        predicted = tf.round(self.preds)

        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.float32)
        labels = tf.cast(self.placeholders['labels'], dtype=tf.float32)
        
        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.count_nonzero(predicted * labels)
        tn = tf.count_nonzero((predicted - 1) * (labels - 1))
        fp = tf.count_nonzero(predicted * (labels - 1))
        fn = tf.count_nonzero((predicted - 1) * labels)

        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        fmeasure = tf.cast(fmeasure, tf.float32)
        return fmeasure
    '''
