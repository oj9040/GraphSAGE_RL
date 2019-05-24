## Advancing GraphSAGE with A Data-driven Node Sampling

#### Authors: [Jihun Oh](http://jihunoh.weebly.com) (oj9040@gmail.com, jihun2331.oh@samsung.com), [Kyunghyun Cho](http://www.kyunghyuncho.me) (kyunghyun.cho@nyu.edu), [Joan Bruna](https://cims.nyu.edu/~bruna/) (bruna@cims.nyu.edu)

### Overview
This work is an improved version of [GraphSAGE](https://github.com/williamleif/GraphSAGE).

As an efficient and scalable graph neural network, GraphSAGE has enabled an inductive
capability for inferring unseen nodes or graphs by aggregating subsampled
local neighborhoods and by learning in a mini-batch gradient descent fashion. The
neighborhood sampling used in GraphSAGE is effective in order to improve computing
and memory efficiency when inferring a batch of target nodes with diverse
degrees in parallel. Despite this advantage, the default uniform sampling suffers
from high variance in training and inference, leading to sub-optimum accuracy.
We propose a new data-driven sampling approach to reason about the real-valued
importance of a neighborhood by a non-linear regressor, and to use the value as a
criterion for subsampling neighborhoods. The regressor is learned using a valuebased
reinforcement learning. The implied importance for each combination of
vertex and neighborhood is inductively extracted from the negative classification
loss output of GraphSAGE. As a result, in an inductive node classification benchmark
using three datasets, our method enhanced the baseline using the uniform
sampling, outperforming recent variants of a graph neural network in accuracy.


### Requirements

Recent versions of TensorFlow, numpy, scipy, sklearn, and networkx are required (but networkx must be <=1.11). You can install all the required packages using the following command:

	$ pip install -r requirements.txt

To guarantee that you have the right package versions, you can use [docker](https://docs.docker.com/) to easily set up a virtual environment. See the Docker subsection below for more info.

#### Docker

If you do not have [docker](https://docs.docker.com/) installed, you will need to do so. (Just click on the preceding link, the installation is pretty painless).  

You can run GraphSage inside a [docker](https://docs.docker.com/) image. After cloning the project, build and run the image as following:

	$ docker build -t graphsage .
	$ docker run -it graphsage bash

or start a Jupyter Notebook instead of bash:

	$ docker run -it -p 8888:8888 graphsage

You can also run the GPU image using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker):

	$ docker build -t graphsage:gpu -f Dockerfile.gpu .
	$ nvidia-docker run -it graphsage:gpu bash	

### Running the code

The example_unsupervised.sh file contains example usages for three dataset (PPI, Reddit, Pubmed) of the code in the supervised classification task.

If your benchmark/task does not require generalizing to unseen data, we recommend you try setting the "--identity_dim" flag to a value in the range [64,256].
This flag will make the model embed unique node ids as attributes, which will increase the runtime and number of parameters but also potentially increase the performance.
Note that you should set this flag and *not* try to pass dense one-hot vectors as features (due to sparsity).
The "dimension" of identity features specifies how many parameters there are per node in the sparse identity-feature lookup table.

*Note:* For the PPI data, and any other multi-ouput dataset that allows individual nodes to belong to multiple classes, it is necessary to set the `--sigmoid` flag during supervised training. By default the model assumes that the dataset is in the "one-hot" categorical setting.


#### Input format
As input, at minimum the code requires that a --train_prefix option is specified which specifies the following data files:

* <train_prefix>-G.json -- A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
* <train_prefix>-id_map.json -- A json-stored dictionary mapping the graph node ids to consecutive integers.
* <train_prefix>-class_map.json -- A json-stored dictionary mapping the graph node ids to classes.
* <train_prefix>-feats.npy [optional] --- A numpy-stored array of node features; ordering given by id_map.json. Can be omitted and only identity features will be used.
* <train_prefix>-walks.txt [optional] --- A text file specifying random walk co-occurrences (one pair per line) (*only for unsupervised version of graphsage)

To run the model on a new dataset, you need to make data files in the format described above.
To run random walks for the unsupervised model and to generate the <prefix>-walks.txt file)
you can use the `run_walks` function in `graphsage.utils`.


#### Dataset Download
Below dataset are not included in github due to big size, but can be downloaded from links

PPI (Protein-Protein Interaction)

    $ wget http://snap.stanford.edu/graphsage/ppi.zip

Reddit

    $ wget http://snap.stanford.edu/graphsage/reddit.zip


Pubmed is included in ./data/pubmed folder
For the dataset with different format, convert it using create_Graph_forGraphSAGE.py.


#### Model variants
The user must also specify a --model, the variants of which are described in detail in the paper:
* mean_concat -- GraphSage with mean-concat based aggregator
* mean_add -- GraphSage with mean-add based aggregator (default)
* graphsage_seq -- GraphSage with LSTM-based aggregator
* graphsage_maxpool -- GraphSage with max-pooling aggregator (as described in the NIPS 2017 paper)
* graphsage_meanpool -- GraphSage with mean-pooling aggregator (a variant of the pooling aggregator, where the element-wie mean replaces the element-wise max).
* gcn -- GraphSage with GCN-based aggregator
* n2v -- an implementation of [DeepWalk](https://arxiv.org/abs/1403.6652) (called n2v for short in the code.)

#### Logging directory
Finally, a --base_log_dir should be specified (it defaults to the current directory).
The output of the model and log files will be stored in a subdirectory of the base_log_dir.
The path to the logged data will be of the form `<sup/unsup>-<data_prefix>/graphsage-<model_description>/`.
The supervised model will output F1 scores, while the unsupervised model will train embeddings and store them.
The unsupervised embeddings will be stored in a numpy formated file named val.npy with val.txt specifying the order of embeddings as a per-line list of node ids.
Note that the full log outputs and stored embeddings can be 5-10Gb in size (on the full data when running with the unsupervised variant).


#### Bibliography

@article{oh2019advancing,  
  title={Advancing GraphSAGE with A Data-Driven Node Sampling},  
    author={Oh, Jihun and Cho, Kyunghyun and Bruna, Joan},  
      journal={arXiv preprint arXiv:1904.12935},  
        year={2019}  
        }

