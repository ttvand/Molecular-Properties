from functools import partial
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf
import multiprocessing as mp
import numpy as np
import networkx as nx
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import pickle
import sonnet as snt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
#  from tensorflow.python.keras.utils import losses_utils
import time


data_folder = '/media/tom/cbd_drive/Kaggle/Molecular Properties/Data/'
#data_folder = '/home/tom/Kaggle/Molecular Properties/Data/'

# Dict that maps an atom to a one-hot embedding
ATOM_ONEHOTS = {
    'H': np.array([1, 0, 0, 0, 0]),
    'C': np.array([0, 1, 0, 0, 0]),
    'N': np.array([0, 0, 1, 0, 0]),
    'O': np.array([0, 0, 0, 1, 0]),
    'F': np.array([0, 0, 0, 0, 1]),
    }

# Dict that maps a binding type to a one-hot embedding
BINDING_ONEHOTS = {
    '1JHC': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '1JHN': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '2JHH': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '2JHC': np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '2JHN': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '3JHH': np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '3JHC': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '3JHN': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '1.0CC': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '1.0CF': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '1.0CN': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '1.0CO': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '1.0HO': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '1.0NN': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '1.0NO': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    '1.5CO': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    '2.0CC': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    '2.0CN': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    '2.0CO': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    '2.0NN': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    '2.0NO': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    '3.0CC': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    '3.0CN': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    }


def get_next_ids(df, name, start_id, considered_nexts = 10000):
  sub_df = df[start_id:(start_id+considered_nexts)]
  match_ids = np.where(sub_df.molecule_name == name)[0]
  assert(match_ids.size > 0 and match_ids[-1] < (considered_nexts-1))
  
  return start_id + match_ids[-1] + 1, start_id + match_ids


def graphs_from_nodes_edges_and_globals(
    graph_nodes, graph_edges, graph_globals, target_graph, graph_nx_format):
  num_nodes = graph_nodes.shape[0]
  num_edges = graph_edges.shape[0]
  
  graph_nx = nx.OrderedMultiDiGraph()
  graph_no_nx = [0, [], []]
  
  # Exclude columns from features - either because they are no features or
  # because they are categorical
  exclude_cols = ['molecule_name', 'atom_index', 'atom_index_0',
                  'atom_index_1', 'atom', 'atom_0', 'atom_1', 'type',
                  'bond_type']
  
  # Target columns (they may not all be present)
  main_target_col = 'scalar_coupling_constant'
  target_cols = [main_target_col, 'fc', 'sd', 'pso', 'dso']
  target_cols = list(set(target_cols).intersection(set(graph_edges.columns)))
  # Make sure that scalar_coupling_constant is the first in the list
  if target_cols:
    scc_id = np.where(np.array(target_cols) == main_target_col)[0][0]
    if scc_id > 0:
      target_cols[scc_id] = target_cols[0]
      target_cols[0] = main_target_col
    
  # Globals
  if target_graph:
    global_features = np.array([])
  else:
    global_numeric = graph_globals.drop(exclude_cols, errors='ignore')
    global_features = global_numeric.values
    
  global_features = global_features.astype(np.float)
  if graph_nx_format:
    graph_nx.graph['features'] = global_features
  else:
    graph_no_nx[0] = global_features
  
  # Nodes
  assert(np.all(graph_nodes.atom_index.values == np.arange(num_nodes)))
  graph_node_numeric = graph_nodes.drop(exclude_cols, axis=1, errors='ignore')
  for node_id in range(num_nodes):
    if target_graph:
      node_features = np.array([])
    else:
      atom_onehot = ATOM_ONEHOTS[graph_nodes.atom.values[node_id]]
      node_numeric = graph_node_numeric.iloc[node_id].values
      node_features = np.hstack([atom_onehot, node_numeric])
      
    node_features = node_features.astype(np.float)
    if graph_nx_format:
      graph_nx.add_node(node_id, features=node_features)
    else:
      graph_no_nx[1].append((node_id, node_features))
  
  # Edges
  no_edge_cols = exclude_cols + target_cols
  graph_edge_numeric = graph_edges.drop(no_edge_cols, axis=1, errors='ignore')
  for edge_id in range(num_edges):
    first_node = graph_edges.atom_index_0.values[edge_id]
    second_node = graph_edges.atom_index_1.values[edge_id]
    if target_graph:
      binding_onehot = BINDING_ONEHOTS[graph_edges.bond_type.values[edge_id]]
      edge_type_id = np.where(binding_onehot == 1)[0][0]
      if edge_type_id < 8:
        target_values = graph_edges[target_cols].values[edge_id]
      else:
        target_values = np.empty((len(target_cols)))
        target_values[:] = np.nan
      edge_features = np.hstack([target_values, edge_type_id])
    else:
      atom_onehot_0 = ATOM_ONEHOTS[graph_edges.atom_0.values[edge_id]]
      atom_onehot_1 = ATOM_ONEHOTS[graph_edges.atom_1.values[edge_id]]
      binding_onehot = BINDING_ONEHOTS[graph_edges.bond_type.values[edge_id]]
      edge_numeric = graph_edge_numeric.iloc[edge_id].values
      edge_features = np.hstack([binding_onehot, atom_onehot_0, atom_onehot_1,
                                 edge_numeric])
      
    edge_features = edge_features.astype(np.float)
    if graph_nx_format:
      graph_nx.add_edge(first_node, second_node, features=edge_features)
      graph_nx.add_edge(second_node, first_node, features=edge_features)
    else:
      # Make sure to create a bidirectional edge at graph construction time!!!
      graph_no_nx[2].append((first_node, second_node, edge_features))
    
  return_graph = graph_nx if graph_nx_format else graph_no_nx
  return return_graph


# Load the list of all graphs and generate them if they don't exist yet.
def load_all_graphs(source='train', target_graph=False, recompute_graphs=False,
                    parallel=True, graph_nx_format=False):
  target_ext = '_target' if target_graph else ''
  graph_file = data_folder + source + target_ext + '_graphs.pickle'
  
  if not recompute_graphs and os.path.exists(graph_file):
    with open(graph_file, 'rb') as f:
      return tuple(pickle.load(f))
  else:
    print('Loading source files')
    nodes = pd.read_csv(data_folder + source + '_extended_nodes.csv')
    edges = pd.read_csv(data_folder + source + '_extended_edges.csv')
    globals_ = pd.read_csv(data_folder + source + '_extended_globals.csv')
    
    molecule_names = edges.molecule_name.unique()
    all_graphs = []
    
    # Obtain all node, edge and global ids
    print('Obtain all node, edge and global ids')
    all_nodes = []
    all_edges = []
    all_globals = []
    start_id_nodes = 0
    start_id_edges = 0
    for graph_id, name in enumerate(molecule_names):
      start_id_nodes, node_ids = get_next_ids(nodes, name, start_id_nodes)
      start_id_edges, edge_ids = get_next_ids(edges, name, start_id_edges)
      all_nodes.append(nodes.iloc[node_ids])
      all_edges.append(edges.iloc[edge_ids])
      all_globals.append(globals_.iloc[graph_id])
      
    print('Starting the graph generation')
    if parallel:
      pool = mp.Pool(processes=mp.cpu_count()-1)
      results = [pool.apply_async(
          graphs_from_nodes_edges_and_globals, args=(
              n, e, g, target_graph, graph_nx_format,)) for (
              n, e, g) in zip(all_nodes, all_edges, all_globals)]
      all_graphs = [p.get() for p in results]
    else:
      for i, name in enumerate(molecule_names):
        print('Molecule {} of {}.'.format(i+1, len(molecule_names)))
        all_graphs.append(graphs_from_nodes_edges_and_globals(
            all_nodes[i], all_edges[i], all_globals[i], target_graph,
            graph_nx_format))
      
    # Save the graphs and associated molecule names
    graphs_and_names = [all_graphs, molecule_names.tolist()]
    with open(graph_file, 'wb') as f:
      pickle.dump(graphs_and_names, f, protocol=pickle.HIGHEST_PROTOCOL)
      return tuple(graphs_and_names)
    
    
def generate_graph(raw_graph):
  num_nodes = len(raw_graph[1])
  num_edges = len(raw_graph[2])
  graph_nx = nx.OrderedMultiDiGraph()
  
  # Globals
  graph_nx.graph['features'] = raw_graph[0]
  
  # Nodes
  for node_id in range(num_nodes):
    graph_nx.add_node(raw_graph[1][node_id][0],
                      features=raw_graph[1][node_id][1])
    
  # Edges
  for edge_id in range(num_edges):
    graph_nx.add_edge(raw_graph[2][edge_id][0], raw_graph[2][edge_id][1],
                      features=raw_graph[2][edge_id][2])
    graph_nx.add_edge(raw_graph[2][edge_id][1], raw_graph[2][edge_id][0],
                      features=raw_graph[2][edge_id][2])
    
  return graph_nx
    
    
def generate_networkx_graphs(rand, num_examples, raw_input_graphs,
                             raw_target_graphs, sample_ids, replace):
  """Generate graphs for training."""
  input_graphs = []
  target_graphs = []
  samples = np.random.choice(sample_ids, size=(num_examples), replace=replace)
  for sample_id in samples:
    input_graph = generate_graph(raw_input_graphs[sample_id])
    target_graph = generate_graph(raw_target_graphs[sample_id])
    input_graphs.append(input_graph)
    target_graphs.append(target_graph)
    
  return input_graphs, target_graphs
    
    
def create_placeholders(rand, batch_size, raw_input_graphs, raw_target_graphs):
  """Creates placeholders for the model training and evaluation."""
  # Create some example data for inspecting the vector sizes.
  input_graphs, target_graphs = generate_networkx_graphs(
      rand, batch_size, raw_input_graphs, raw_target_graphs, np.arange(2),
      True)
  input_ph = utils_tf.placeholders_from_networkxs(input_graphs)
  target_ph = utils_tf.placeholders_from_networkxs(target_graphs)
  weight_ph = tf.zeros_like(target_ph.edges[:, :1])
  return input_ph, target_ph, weight_ph


def create_feed_dict(rand, batch_size, raw_input_graphs, raw_target_graphs,
                     input_ph, target_ph, weight_ph, edge_rel_weights,
                     sample_ids, replace):
  """Creates placeholders for the model training and evaluation."""
  # Create some example data for inspecting the vector sizes.
  inputs, targets = generate_networkx_graphs(
      rand, batch_size, raw_input_graphs, raw_target_graphs, sample_ids,
      replace)
  input_graphs = utils_np.networkxs_to_graphs_tuple(inputs)
  target_graphs = utils_np.networkxs_to_graphs_tuple(targets)
  graph_edge_weights = edge_rel_weights[target_graphs[1][:, -1].astype(np.int)]
  norm_graph_edge_weights = graph_edge_weights/(
      graph_edge_weights[graph_edge_weights>0].mean())
  feed_dict = {input_ph: input_graphs, target_ph: target_graphs,
               weight_ph: norm_graph_edge_weights}
  return feed_dict


## Adapted from https://stackoverflow.com/questions/56099314/what-is-a-replacement-for-tf-losses-absolute-difference
#def absolute_difference_ignore_nans(labels, predictions, weights=1.0,
#                                    reduction='mean'):
#  if reduction == 'mean':
#    reduction_fn = tf.reduce_mean
#  elif reduction == 'sum':
#    reduction_fn = tf.reduce_sum
#  else:
#    # You could add more reductions
#    pass
#  labels = tf.cast(labels, tf.float32)
#  predictions = tf.cast(predictions, tf.float32)
#  losses = tf.abs(tf.subtract(predictions, labels))
#  
#  
#  
#  weights = tf.cast(tf.convert_to_tensor(weights), tf.float32)
#  import pdb; pdb.set_trace()
#  res = losses_utils.compute_weighted_loss(
#      losses, weights, reduction=tf.keras.losses.Reduction.NONE)
#
#  return reduction_fn(res, axis=None)


def create_loss_ops(target_op, output_ops, weight_ph):
  # Ignore nans
  labels = target_op.edges[:, :1]
  no_nan_edge_labels = tf.where(
      tf.math.is_nan(labels), tf.zeros_like(labels), labels)
  no_nan_edge_outputs = [tf.where(tf.math.is_nan(labels),
                                  tf.zeros_like(labels),
                                  o.edges) for o in output_ops]
  
  loss_ops = [
      tf.losses.absolute_difference(no_nan_edge_labels, no_nan_output,
                                    weights=weight_ph)
#      tf.losses.mean_squared_error(target_op.edges, output_op.edges)
      for no_nan_output in no_nan_edge_outputs
  ]
  return loss_ops


def make_all_runnable_in_session(*args):
  """Lets an iterable of TF graphs be output from a session as NP graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]


def batch_id_epoch_generator(sample_ids, batch_size):
  num_batches_per_epoch = sample_ids.size // batch_size
  while True:
    permuted_ids = np.random.permutation(sample_ids)
    for i in range(num_batches_per_epoch):
      yield permuted_ids[(i*batch_size):((i+1)*batch_size)]
      
      
def make_mlp_model(latent_size, num_layers):
  """Instantiates a new MLP, followed by LayerNorm.
  The parameters of each new MLP are not shared with others generated by
  this function.
  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  return snt.Sequential([
      snt.nets.MLP([latent_size] * num_layers, activate_final=True),
      snt.LayerNorm()
  ])


class MLPGraphIndependent(snt.AbstractModule):
  """GraphIndependent with MLP edge, node, and global models."""

  def __init__(self, latent_size=16, num_layers=2, double_edge=False,
               name="MLPGraphIndependent"):
    super(MLPGraphIndependent, self).__init__(name=name)
    edge_size = 2*latent_size if double_edge else latent_size
    node_size = 1 if double_edge else latent_size
    global_size = 1 if double_edge else latent_size
    with self._enter_variable_scope():
      self._network = modules.GraphIndependent(
          edge_model_fn=partial(make_mlp_model, latent_size=edge_size,
                                num_layers=num_layers),
          node_model_fn=partial(make_mlp_model, latent_size=node_size,
                                num_layers=num_layers),
          global_model_fn=partial(make_mlp_model, latent_size=global_size,
                                num_layers=num_layers))

  def _build(self, inputs):
    return self._network(inputs)


class MLPGraphNetwork(snt.AbstractModule):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self, latent_size=16, num_layers=2, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.GraphNetwork(
          partial(make_mlp_model, latent_size=latent_size,
                  num_layers=num_layers),
          partial(make_mlp_model, latent_size=latent_size,
                  num_layers=num_layers),
          partial(make_mlp_model, latent_size=latent_size,
                  num_layers=num_layers))

  def _build(self, inputs):
    return self._network(inputs)


class MyEncodeProcessDecode(snt.AbstractModule):
  # Note: all 3*3 MLPs are currently of the same dimensions, can be decoupled!!
  
  """Full encode-process-decode model.
  The model we explore includes three components:
  - An "Encoder" graph net, which independently encodes the edge, node, and
    global attributes (does not compute relations etc.).
  - A "Core" graph net, which performs N rounds of processing (message-passing)
    steps. The input to the Core is the concatenation of the Encoder's output
    and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
    the processing step).
  - A "Decoder" graph net, which independently decodes the edge, node, and
    global attributes (does not compute relations etc.), on each message-passing
    step.
                      Hidden(t)   Hidden(t+1)
                         |            ^
            *---------*  |  *------*  |  *---------*
            |         |  |  |      |  |  |         |
  Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
            |         |---->|      |     |         |
            *---------*     *------*     *---------*
  """

  def __init__(self,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               latent_size=16,
               num_layers=2,
               name="MyEncodeProcessDecode"):
    super(MyEncodeProcessDecode, self).__init__(name=name)
    self._encoder = MLPGraphIndependent(latent_size, num_layers)
    self._core = MLPGraphNetwork(latent_size, num_layers)
    self._decoder = MLPGraphIndependent(latent_size, num_layers,
                                        double_edge=True)
    # Transforms the outputs into the appropriate shapes.
    if edge_output_size is None:
      edge_fn = None
    else:
      edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
    if node_output_size is None:
      node_fn = None
    else:
      node_fn = lambda: snt.Linear(node_output_size, name="node_output")
    if global_output_size is None:
      global_fn = None
    else:
      global_fn = lambda: snt.Linear(global_output_size, name="global_output")
    with self._enter_variable_scope():
      self._output_transform = modules.GraphIndependent(edge_fn, node_fn,
                                                        global_fn)

  def _build(self, input_op, num_processing_steps):
    latent = self._encoder(input_op)
    latent0 = latent
    output_ops = []
    for _ in range(num_processing_steps):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = self._core(core_input)
      decoded_op = self._decoder(latent)
      output_ops.append(self._output_transform(decoded_op))
    return output_ops


def train_model(hyperpars, train_graphs, train_target_graphs, train_ids,
                valid_ids, model_save_path):
  # 1) Set up model training and evaluation  
  
  # The model we explore includes three components:
  # - An "Encoder" graph net, which independently encodes the edge, node, and
  #   global attributes (does not compute relations etc.).
  # - A "Core" graph net, which performs N rounds of processing (message-passing)
  #   steps. The input to the Core is the concatenation of the Encoder's output
  #   and the previous output of the Core (labeled "Hidden(t)" below, where "t"
  #   is the processing step).
  # - A "Decoder" graph net, which independently decodes the edge, node, and
  #   global attributes (does not compute relations etc.), on each
  #   message-passing step.
  #
  #                     Hidden(t)   Hidden(t+1)
  #                        |            ^
  #           *---------*  |  *------*  |  *---------*
  #           |         |  |  |      |  |  |         |
  # Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
  #           |         |---->|      |     |         |
  #           *---------*     *------*     *---------*
  #
  # The model is trained by supervised learning. 
  #
  # The training loss is computed on the output of each processing step. The
  # reason for this is to encourage the model to try to solve the problem in as
  # few steps as possible. It also helps make the output of intermediate steps
  # more interpretable.
  
  
  # Data. Input and target placeholders.
  tf.reset_default_graph()
  rand = np.random.RandomState(seed=hyperpars['seed'])
  input_ph, target_ph, weight_ph = create_placeholders(
      rand, hyperpars['batch_size_train'], train_graphs, train_target_graphs)
  
  # Connect the data to the model.
  # Instantiate the model.
  model = MyEncodeProcessDecode(edge_output_size=1,
                                latent_size=hyperpars['latent_size'],
                                num_layers=hyperpars['num_layers'])
  # A list of outputs, one per processing step.
  output_ops_train = model(input_ph, hyperpars['num_processing_steps'])
  output_ops_valid = model(input_ph, hyperpars['num_processing_steps'])
  
  # Training loss.
  loss_ops_train = create_loss_ops(target_ph, output_ops_train, weight_ph)
  # Loss across processing steps.
  loss_op_train = sum(loss_ops_train) / hyperpars['num_processing_steps']
  
  # Validation loss.
  loss_ops_valid = create_loss_ops(target_ph, output_ops_valid, weight_ph)
  loss_op_valid = loss_ops_valid[-1]  # Loss from final processing step.
  
  # Optimizer.
  learning_rate = hyperpars['learning_rate']
  optimizer = tf.train.AdamOptimizer(learning_rate)
  step_op = optimizer.minimize(loss_op_train)
  
  # Lets an iterable of TF graphs be output from a session as NP graphs.
  input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)
  
  
  # 2) Create session, saver and initialize the global variables
  sess = tf.Session()
  saver = tf.train.Saver()
  sess.run(tf.global_variables_initializer())
  
  # 3) Run training
  print('\n# (iteration number), T (elapsed seconds), '
        'Ltrain (training loss), Lvalid (validation loss), '
        'Svalid (validation score)'
        )
  
  start_time = time.time()
  last_log_time = start_time
  logged_iterations = []
  losses_train = []
  losses_valid = []
  best_validation_loss = float('inf')
  train_losses_since_last_log = []
  batch_epoch_generator = batch_id_epoch_generator(
      train_ids, hyperpars['batch_size_train'])
  edge_rel_weights = np.ones((23, 1)) # 23 different bond types
  edge_rel_weights[8:, 0] = -99999
  for iteration in range(hyperpars['num_training_iterations']):
    batch_train_ids = next(batch_epoch_generator)
    feed_dict = create_feed_dict(
        rand, hyperpars['batch_size_train'], train_graphs, train_target_graphs,
        input_ph, target_ph, weight_ph, edge_rel_weights, batch_train_ids,
        replace=False)
    train_values = sess.run({
        'step': step_op,
        'target': target_ph,
        'loss': loss_op_train,
        'outputs': output_ops_train,
    }, feed_dict=feed_dict)
    train_losses_since_last_log.append(train_values['loss'])
#    print(train_values['loss'])
#    [n.name for n in tf.trainable_variables() if not 'adam' in n.name.lower()]
#    var = [v for v in tf.trainable_variables() if v.name == "MLPGraphIndependent/graph_independent/edge_model/mlp/linear_0/w:0"][0]
#    var_vals = sess.run(var)
#    print(var_vals.sum())
    
    the_time = time.time()
    elapsed_since_last_log = the_time - last_log_time
    if (elapsed_since_last_log > hyperpars['log_every_seconds']) or (
        iteration == (hyperpars['num_training_iterations']-1)):
#      before_validation_time = time.time()
      last_log_time = the_time
      feed_dict = create_feed_dict(
          rand, hyperpars['batch_size_valid_train'], train_graphs,
          train_target_graphs, input_ph, target_ph, weight_ph,
          edge_rel_weights, valid_ids, replace=False)
      valid_values = sess.run({
          'target': target_ph,
          'loss': loss_op_valid,
          'outputs': output_ops_valid
      }, feed_dict=feed_dict)
      validation_loss = valid_values['loss']
    
      targets = valid_values['target'][1][:, :1]
      binding_types = valid_values['target'][1][:, -1:]
      predictions = valid_values['outputs'][-1][1]
      errors = targets - predictions
      
      scores = []
      for binding_id in range(8):
        type_ids = binding_types == binding_id
        scores.append(np.log(np.mean(np.abs(errors[type_ids]))))
      validation_score = np.array(scores).mean()
      if hyperpars['inverse_relative_error_weights']:
        edge_rel_weights[:8, 0] = np.exp(-1*np.array(scores))
    
      elapsed = time.time() - start_time
      average_train_loss = np.array(train_losses_since_last_log).mean()
      train_losses_since_last_log = []
      losses_train.append(average_train_loss)
      losses_valid.append(validation_loss)
      logged_iterations.append(iteration)
      progress_format = '# {:05d}, T {:.1f}, Ltrain {:.4f}, ' + (
      'Lvalid {:.4f}, Svalid {:.4f}')
      print(progress_format.format(iteration+1, elapsed, average_train_loss,
                                   validation_loss, validation_score))
  
      # Save the variables to disk if the validation loss has improved.
      if validation_loss < best_validation_loss:
        saver.save(sess, model_save_path)
#        print('Elapsed validation time: {}'.format(
#            time.time()-before_validation_time))
        best_validation_loss = validation_loss
        

def validate_model(hyperpars, train_graphs, train_target_graphs, valid_ids,
                   model_save_path):
  # Data. Input and target placeholders.
  tf.reset_default_graph()
  rand = np.random.RandomState(seed=hyperpars['seed'])
  input_ph, target_ph, weight_ph = create_placeholders(
      rand, hyperpars['batch_size_train'], train_graphs, train_target_graphs)
  
  # Connect the data to the model.
  # Instantiate the model.
  model = MyEncodeProcessDecode(edge_output_size=1,
                                latent_size=hyperpars['latent_size'],
                                num_layers=hyperpars['num_layers'])
  # A list of outputs, one per processing step.
  output_ops_valid = model(input_ph, hyperpars['num_processing_steps'])
  
  # Validation loss.
  loss_ops_valid = create_loss_ops(target_ph, output_ops_valid, weight_ph)
  loss_op_valid = loss_ops_valid[-1]  # Loss from final processing step.
  
  # Lets an iterable of TF graphs be output from a session as NP graphs.
  input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)
  
  
  # 2) Create saver and session.
  sess = tf.Session()
  saver = tf.train.Saver()
  
  # 3) Run validation
  saver.restore(sess, model_save_path)
  edge_rel_weights = np.ones((23, 1)) # 23 different bond types
  edge_rel_weights[8:, 0] = -99999
  feed_dict = create_feed_dict(
      rand, hyperpars['batch_size_valid'], train_graphs,
      train_target_graphs, input_ph, target_ph, weight_ph, edge_rel_weights,
      valid_ids, replace=False)
  valid_values = sess.run({
      'target': target_ph,
      'loss': loss_op_valid,
      'outputs': output_ops_valid,
  }, feed_dict=feed_dict)
  
  validation_loss = valid_values['loss']
  print('Validation loss (MAE): {:.4f}'.format(validation_loss))
  
  # Analyze the errors by binding type
  targets = valid_values['target'][1][:, :1]
  binding_types = valid_values['target'][1][:, -1:]
  predictions = valid_values['outputs'][-1][1]
  errors = targets - predictions
  
#  target_pred_errors = np.hstack([targets, predictions, errors, binding_types])
  scores = []
  for binding_id in range(8):
    type_ids = binding_types == binding_id
    scores.append(np.log(np.mean(np.abs(errors[type_ids]))))
  validation_score = np.array(scores).mean()
  print('Validation score: {:.4f}'.format(validation_score))
  
  return validation_score, validation_loss


#load_all_graphs('test', recompute_graphs=True)
#load_all_graphs('train', recompute_graphs=True)
#load_all_graphs('train', target_graph=True, recompute_graphs=True)
