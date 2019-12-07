from functools import partial
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf
import matplotlib.pyplot as plt
import math
import models
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
from tensorflow.keras import backend as K
import time


data_folder = '/media/tom/cbd_drive/Kaggle/Molecular Properties/Data/'
#data_folder = '/home/tom/Kaggle/Molecular-Properties/Data/'

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

# Dict that maps a 3J connection atom to a one-hot embedding
ATOM_3J_ONEHOTS = {
    'C': np.array([1, 0, 0]),
    'N': np.array([0, 1, 0]),
    'O': np.array([0, 0, 1]),
    'None': np.array([0, 0, 0]),
    }


def get_next_ids(df, name, start_id, considered_nexts = 10000):
  sub_df = df[start_id:(start_id+considered_nexts)]
  match_ids = np.where(sub_df.molecule_name == name)[0]
  assert(match_ids.size > 0 and match_ids[-1] < (considered_nexts-1))
  
  return start_id + match_ids[-1] + 1, start_id + match_ids


def get_other_edge_ids(cat_sizes_pairs, numeric_colnames):
  cat_sizes, cat_pairs = cat_sizes_pairs
  num_categorical = np.array(cat_sizes).sum()
  num_numeric = len(numeric_colnames)
  num_features = num_categorical + num_numeric
  other_permutation = np.arange(num_features)
  
  # Add categorical permutations
  cat_counter = 0
  for (i, s) in enumerate(cat_sizes):
    for first, second in cat_pairs:
      if i == first and (cat_sizes[first] == cat_sizes[second]):
        first_ids = cat_counter + np.arange(s)
        second_ids = cat_counter + s + np.arange(s)
        tmp = other_permutation[first_ids]
        other_permutation[first_ids] = other_permutation[second_ids]
        other_permutation[second_ids] = tmp
        
    cat_counter += s
  
  # Add numerical permutations
  numeric_pairs = [
      ('atom_0_couples_count', 'atom_1_couples_count'),
      ('atom_0_couples_count_direct', 'atom_1_couples_count_direct'),
      ('molecule_atom_index_0_dist_mean', 'molecule_atom_index_1_dist_mean'),
      ('molecule_atom_index_0_dist_max', 'molecule_atom_index_1_dist_max'),
      ('molecule_atom_index_0_dist_min', 'molecule_atom_index_1_dist_min'),
      ('molecule_atom_index_0_dist_std', 'molecule_atom_index_1_dist_std'),
      ('molecule_atom_index_0_dist_mean_diff', 'molecule_atom_index_1_dist_mean_diff'),
      ('molecule_atom_index_0_dist_mean_div', 'molecule_atom_index_1_dist_mean_div'),
      ('molecule_atom_index_0_dist_max_diff', 'molecule_atom_index_1_dist_max_diff'),
      ('molecule_atom_index_0_dist_max_div', 'molecule_atom_index_1_dist_max_div'),
      ('molecule_atom_index_0_dist_min_diff', 'molecule_atom_index_1_dist_min_diff'),
      ('molecule_atom_index_0_dist_min_div', 'molecule_atom_index_1_dist_min_div'),
      ('molecule_atom_index_0_dist_std_diff', 'molecule_atom_index_1_dist_std_diff'),
      ('molecule_atom_index_0_dist_std_div', 'molecule_atom_index_1_dist_std_div'),
      ('molecule_atom_0_dist_mean', 'molecule_atom_1_dist_mean'),
      ('molecule_atom_0_dist_min', 'molecule_atom_1_dist_min'),
      ('molecule_atom_0_dist_std', 'molecule_atom_1_dist_std'),
      ('molecule_atom_0_dist_min_diff', 'molecule_atom_1_dist_min_diff'),
      ('molecule_atom_0_dist_min_div', 'molecule_atom_1_dist_min_div'),
      ('molecule_atom_0_dist_std_diff', 'molecule_atom_1_dist_std_diff'),
      ('distance_0', 'distance_1'),
      ('cos_0', 'cos_1'),
      ('atom_0_interm_0_interm_1_cosine', 'interm_0_interm_1_atom_1_cosine'),
      ]
  for first, second in numeric_pairs:
    first_numeric_id = np.where(numeric_colnames==first)[0][0]
    second_numeric_id = np.where(numeric_colnames==second)[0][0]
    assert(first_numeric_id != second_numeric_id)
    
    tmp = other_permutation[first_numeric_id + num_categorical]
    other_permutation[first_numeric_id+num_categorical] = other_permutation[
        second_numeric_id+num_categorical]
    other_permutation[second_numeric_id+num_categorical] = tmp
  
  return other_permutation


def graphs_from_nodes_edges_and_globals(
    graph_nodes, graph_edges, graph_globals, only_edge, target_graph,
    graph_nx_format):
  num_nodes = graph_nodes.shape[0]
  num_edges = graph_edges.shape[0]
  
  graph_nx = nx.OrderedMultiDiGraph()
  graph_no_nx = [0, [], []]
  
  # Exclude columns from features - either because they are no features or
  # because they are categorical
  exclude_cols = ['molecule_name', 'atom_index', 'atom_index_0',
                  'atom_index_1', 'atom', 'atom_0', 'atom_1', 'type',
                  'bond_type', 'interm_0_atom', 'interm_1_atom']
  
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
      other_edge_permutation = np.zeros(())
    else:
      atom_onehot_0 = ATOM_ONEHOTS[graph_edges.atom_0.values[edge_id]]
      atom_onehot_1 = ATOM_ONEHOTS[graph_edges.atom_1.values[edge_id]]
      binding_onehot = BINDING_ONEHOTS[graph_edges.bond_type.values[edge_id]]
      connect_0_3J_onehot = ATOM_3J_ONEHOTS[
          graph_edges.interm_0_atom.values[edge_id]]
      connect_1_3J_onehot = ATOM_3J_ONEHOTS[
          graph_edges.interm_1_atom.values[edge_id]]
      edge_numeric = graph_edge_numeric.iloc[edge_id].values
      edge_features = np.hstack([binding_onehot, atom_onehot_0, atom_onehot_1,
                                 connect_0_3J_onehot, connect_1_3J_onehot,
                                 edge_numeric])
      
      if edge_id == 0:
        other_edge_permutation = get_other_edge_ids(([
            binding_onehot.size, atom_onehot_0.size, atom_onehot_1.size,
            connect_0_3J_onehot.size, connect_1_3J_onehot.size], [
                (1, 2), (3, 4)]), graph_edge_numeric.columns)
      
    edge_features = edge_features.astype(np.float)
    if graph_nx_format:
      graph_nx.add_edge(first_node, second_node, features=edge_features)
      graph_nx.add_edge(second_node, first_node, features=edge_features)
    else:
      # Make sure to create a bidirectional edge at graph construction time!!!
      graph_no_nx[2].append((first_node, second_node, edge_features))
    
  return_graph = graph_nx if graph_nx_format else graph_no_nx
  if only_edge:
    return_graph = convert_to_edge_features(return_graph)
  return (return_graph, other_edge_permutation)


def convert_to_edge_features(graph_features):
  global_features = graph_features[0]
  node_details = graph_features[1]
  edge_details  = graph_features[2]
  
  node_ids = np.array([n[0] for n in node_details])
  assert np.all(node_ids == np.arange(len(node_details)))
  node_features = np.stack([n[1] for n in node_details])
  edge_first_ids = np.array([e[0] for e in edge_details])
  edge_second_ids = np.stack([e[1] for e in edge_details])
  edge_features = np.stack([e[2] for e in edge_details])
  first_node_edge_features = node_features[edge_first_ids]
  second_node_edge_features = node_features[edge_second_ids]
  
  all_edge_features = np.concatenate([
      np.tile(global_features, [edge_features.shape[0], 1]),
      first_node_edge_features,
      second_node_edge_features,
      edge_features,
      ], axis=1)
        
  return all_edge_features


# Load the list of all graphs and generate them if they don't exist yet.
def load_all_graphs(source='train', target_graph=False, recompute_graphs=False,
                    parallel=True, graph_nx_format=False,
                    only_edge=False):
  target_ext = '_target' if target_graph else ''
  only_edge_ext = '_only_edge' if only_edge else ''
  graph_file = data_folder + source + target_ext + only_edge_ext + (
      '_graphs.pickle')
  
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
              n, e, g, only_edge, target_graph,
              graph_nx_format,)) for (
              n, e, g) in zip(all_nodes, all_edges, all_globals)]
      all_graphs = [p.get() for p in results]
    else:
      for i, name in enumerate(molecule_names):
        print('Molecule {} of {}.'.format(i+1, len(molecule_names)))
        all_graphs.append(graphs_from_nodes_edges_and_globals(
            all_nodes[i], all_edges[i], all_globals[i], only_edge,
            target_graph, graph_nx_format))
      
    # Save the graphs and associated molecule names
    all_graphs, edge_permutations = zip(*all_graphs)
    import pdb; pdb.set_trace()
    graphs_permutations_and_names = [all_graphs, edge_permutations[0],
                                     molecule_names.tolist()]
    with open(graph_file, 'wb') as f:
      pickle.dump(graphs_permutations_and_names, f,
                  protocol=pickle.HIGHEST_PROTOCOL)
    
    return tuple(graphs_permutations_and_names)
    
    
def generate_graph(raw_graph, edge_permutations, target_graph):
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
    if target_graph:
      graph_nx.add_edge(raw_graph[2][edge_id][1], raw_graph[2][edge_id][0],
                        features=raw_graph[2][edge_id][2])
    else:
      graph_nx.add_edge(raw_graph[2][edge_id][1], raw_graph[2][edge_id][0],
                        features=raw_graph[2][edge_id][2][edge_permutations])
    
  return graph_nx
    
    
def generate_networkx_graphs(rand, num_examples, raw_input_graphs,
                             raw_target_graphs, edge_permutations, sample_ids,
                             replace):
  """Generate graphs for training."""
  input_graphs = []
  target_graphs = []
  samples = np.random.choice(sample_ids, size=(num_examples), replace=replace)
  for sample_id in samples:
    input_graph = generate_graph(raw_input_graphs[sample_id],
                                 edge_permutations, target_graph=False)
    target_graph = generate_graph(raw_target_graphs[sample_id],
                                  edge_permutations, target_graph=True)
    input_graphs.append(input_graph)
    target_graphs.append(target_graph)
    
  return input_graphs, target_graphs
    
    
def create_placeholders(rand, batch_size, raw_input_graphs, raw_target_graphs,
                        edge_permutations):
  """Creates placeholders for the model training and evaluation."""
  # Create some example data for inspecting the vector sizes.
  input_graphs, target_graphs = generate_networkx_graphs(
      rand, batch_size, raw_input_graphs, raw_target_graphs, edge_permutations,
      np.arange(2), True)
  input_ph = utils_tf.placeholders_from_networkxs(input_graphs)
  target_ph = utils_tf.placeholders_from_networkxs(target_graphs)
  weight_ph = tf.zeros_like(target_ph.edges[:, :1])
  return input_ph, target_ph, weight_ph


def create_feed_dict(rand, batch_size, raw_input_graphs, raw_target_graphs,
                     edge_permutations, input_ph, target_ph, weight_ph,
                     edge_rel_weights, sample_ids, replace):
  """Creates placeholders for the model training and evaluation."""
  # Create some example data for inspecting the vector sizes.
  inputs, targets = generate_networkx_graphs(
      rand, batch_size, raw_input_graphs, raw_target_graphs, edge_permutations,
      sample_ids, replace)
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
      
      
class MyMultiMlp(snt.AbstractModule):
  """Docstring for MyMultiMlp."""
  def __init__(self, latent_size, num_layers, layer_norm, num_separate_outputs,
               name="MyMultiMlp"):
    super(MyMultiMlp, self).__init__(name=name)
    self._latent_size = latent_size
    self._num_layers = num_layers
    self._layer_norm = layer_norm
    self._num_separate_outputs = num_separate_outputs
    
  def _build(self, inputs):
    """Compute output Tensor from input Tensor."""
#    import pdb; pdb.set_trace()
    mlp_outputs = []
    for i in range(self._num_separate_outputs):
      mlp_name = 'edge_output_mlp_' + str(i)
      core_mlp = snt.nets.MLP([self._latent_size] * self._num_layers,
                              activate_final=True, name=mlp_name)
      mlp_outputs.append(core_mlp(inputs[:, 8:]))
      
    combined = tf.stack(mlp_outputs, 1)*tf.expand_dims(inputs[:, :8], -1)
    selected = tf.reduce_sum(combined, 1)
    if self._layer_norm:
      selected = snt.LayerNorm()(selected)
    
    return selected
      
      
def make_mlp_model(latent_size, num_layers, layer_norm=True,
                   separate_output=False, num_separate_outputs=8):
  """Instantiates a new MLP, followed by LayerNorm.
  The parameters of each new MLP are not shared with others generated by
  this function.
  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  if separate_output:
    return MyMultiMlp(latent_size, num_layers, layer_norm,
                      num_separate_outputs)
  else:
    layers = [snt.nets.MLP([latent_size] * num_layers, activate_final=True)]
    layers = layers + [snt.LayerNorm()] if layer_norm else layers
    return snt.Sequential(layers)


class MLPGraphIndependent(snt.AbstractModule):
  """GraphIndependent with MLP edge, node, and global models."""

  def __init__(self, latent_size=16, num_layers=2,
               output_independent=False, separate_edge_output=False,
               edge_output_layer_norm=False,
               name="MLPGraphIndependent"):
    super(MLPGraphIndependent, self).__init__(name=name)
    edge_size = 2*latent_size if output_independent else latent_size
    edge_size = edge_size // 4 if separate_edge_output else edge_size
    node_size = 1 if output_independent else latent_size
    global_size = 1 if output_independent else latent_size
    layer_norm = not output_independent or edge_output_layer_norm
    with self._enter_variable_scope():
      self._network = modules.GraphIndependent(
          edge_model_fn=partial(make_mlp_model, latent_size=edge_size,
                                num_layers=num_layers,
                                layer_norm=layer_norm,
                                separate_output=separate_edge_output),
          node_model_fn=partial(make_mlp_model, latent_size=node_size,
                                num_layers=num_layers),
          global_model_fn=partial(make_mlp_model, latent_size=global_size,
                                num_layers=num_layers))

  def _build(self, inputs):
    return self._network(inputs)


class MLPGraphNetwork(snt.AbstractModule):
  """GraphNetwork with MLP edge, node, and global models."""
  # Core of the architecture

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
  # Note: all 3*3 MLPs can be of different dimensions!!
  
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
               separate_edge_output=False,
               edge_output_layer_norm=False,
               skip_encoder_decoder=False,
               name="MyEncodeProcessDecode"):
    super(MyEncodeProcessDecode, self).__init__(name=name)
    self._encoder = MLPGraphIndependent(latent_size, num_layers)
    self._core = MLPGraphNetwork(latent_size, num_layers)
    self._edge_type_concat = separate_edge_output
    self._decoder = MLPGraphIndependent(
        latent_size, num_layers, output_independent=True,
        separate_edge_output=separate_edge_output,
        edge_output_layer_norm=edge_output_layer_norm)
    self._skip_encoder_decoder = skip_encoder_decoder
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
      if self._skip_encoder_decoder:
        decoder_input = utils_tf.concat([latent0, latent], axis=1)
      else:
        decoder_input = latent
      if self._edge_type_concat:
        decoder_input = decoder_input._replace(
            edges=tf.concat([input_op.edges[:, :8], decoder_input.edges],
                            axis=1))
      decoded_op = self._decoder(decoder_input)
      output_ops.append(self._output_transform(decoded_op))
    return output_ops


def train_model(hyperpars, train_graphs, train_target_graphs,
                edge_permutations, train_ids, valid_ids, model_save_path):
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
      rand, hyperpars['batch_size_train'], train_graphs, train_target_graphs,
      edge_permutations)
  
  # Connect the data to the model.
  # Instantiate the model.
  model = MyEncodeProcessDecode(
      edge_output_size=1,
      latent_size=hyperpars['latent_size'],
      num_layers=hyperpars['num_layers'],
      separate_edge_output=hyperpars['separate_edge_output'],
      edge_output_layer_norm=hyperpars['edge_output_layer_norm'],
      skip_encoder_decoder=hyperpars['skip_connection_encoder_decoder'])
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
  for iteration in range(int(hyperpars['num_training_iterations'])):
    batch_train_ids = next(batch_epoch_generator)
    feed_dict = create_feed_dict(
        rand, hyperpars['batch_size_train'], train_graphs, train_target_graphs,
        edge_permutations, input_ph, target_ph, weight_ph, edge_rel_weights,
        batch_train_ids, replace=False)
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
#    for v in tf.trainable_variables():
#      print(v.name)
#    import pdb; pdb.set_trace()
    
    the_time = time.time()
    elapsed_since_last_log = the_time - last_log_time
    if (elapsed_since_last_log > hyperpars['log_every_seconds']) or (
        iteration == (hyperpars['num_training_iterations']-1)):
#      before_validation_time = time.time()
      last_log_time = the_time
      feed_dict = create_feed_dict(
          rand, hyperpars['batch_size_valid_train'], train_graphs,
          train_target_graphs, edge_permutations, input_ph, target_ph,
          weight_ph, edge_rel_weights, valid_ids, replace=False)
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
       
      # Exit if the time budget is exceeded
      if elapsed > hyperpars['max_train_seconds']:
        print('Exiting because the train budget is exceeded')
        break
        

def validate_model(hyperpars, train_graphs, train_target_graphs,
                   edge_permutations, valid_ids, model_save_path):
  # Data. Input and target placeholders.
  tf.reset_default_graph()
  rand = np.random.RandomState(seed=hyperpars['seed'])
  input_ph, target_ph, weight_ph = create_placeholders(
      rand, hyperpars['batch_size_train'], train_graphs, train_target_graphs,
      edge_permutations)
  
  # Connect the data to the model.
  # Instantiate the model.
  model = MyEncodeProcessDecode(
      edge_output_size=1,
      latent_size=hyperpars['latent_size'],
      num_layers=hyperpars['num_layers'],
      separate_edge_output=hyperpars['separate_edge_output'],
      edge_output_layer_norm=hyperpars['edge_output_layer_norm'],
      skip_encoder_decoder=hyperpars['skip_connection_encoder_decoder'])
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
      rand, hyperpars['batch_size_valid'], train_graphs, train_target_graphs,
      edge_permutations, input_ph, target_ph, weight_ph, edge_rel_weights,
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


def masked_mae(y, p, mask_val):
  mask = K.cast(K.not_equal(y, mask_val), K.floatx())
  return tf.compat.v2.losses.mae(y*mask, p*mask)


def make_masked_mae(training_config):
  def loss(y, p):
    return masked_mae(y, p, mask_val=training_config['nan_coding_value'])

  return loss


# Custom Keras callback for plotting learning progress
class PlotLosses(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.i = 0
    self.x = []
    self.val_losses = []
    self.fig = plt.figure()
    self.logs = []
    
    loss_extensions = [
        '', # The no extension loss represents the total loss.
        'val', # Validation loss
#        
#        'segmentation_pred', # Train segmentation loss
#        'val_segmentation_pred', # Validation segmentation loss
        ]
    
    self.best_loss_key = 'loss'
    self.loss_keys = [e + ('_' if e else '') + 'loss' for e in loss_extensions]

  def on_epoch_end(self, epoch, logs={}):
    # Drop absent loss keys on the first epoch end
    if self.i == 0:
      self.loss_keys = list(set(self.loss_keys).intersection(set(logs.keys())))
      self.losses = {k: [] for k in self.loss_keys}
    
    self.logs.append(logs)
    self.x.append(self.i)
    for k in self.loss_keys:
      self.losses[k].append(logs.get(k))
    self.i += 1
    
    best_loss = np.repeat(np.array(self.losses[self.best_loss_key]).min(),
                              self.i).tolist()
    best_id = (1+np.repeat(
        np.array(self.losses[self.best_loss_key]).argmin(), 2)).tolist()
    for k in self.loss_keys:
      plt.plot([1+x for x in self.x], self.losses[k], label=k)
    all_losses = np.array(list(self.losses.values())).flatten()
    if len(self.losses) >= 1:
      plt.plot([1+x for x in self.x], best_loss, linestyle='--', color='r',
               label='')
      plt.plot(best_id, [min(all_losses) - 0.1, best_loss[0]],
               linestyle='--', color='r', label='')
    plt.ylim(0, max(all_losses)*1.02)
    plt.legend()
    plt.show()
    
    
def generator_batch(features, targets, data_ids, hyperpars, batch_size,
                    all_targets=False):
  num_molecules = data_ids.size
  max_edges = hyperpars['max_edges']
  num_edge_features = hyperpars['num_edge_features']
  padded_num_edge_features = hyperpars['padded_num_edge_features']
  num_batches_per_epoch = math.ceil(data_ids.size/batch_size)
  nan_coding_value = hyperpars['nan_coding_value']
  auxiliary_losses = hyperpars['auxiliary_losses']
  transformer_depth = hyperpars['transformer_depth']
  
  batch_in_epoch = 0
  while True:
    if batch_in_epoch == 0:
      shuffled_data_ids = np.random.permutation(data_ids)
    
    # Generate a batch of data
    batch_start_id = batch_in_epoch*batch_size
    batch_end_id = min(num_molecules, (batch_in_epoch+1)*batch_size)
    this_batch_size = batch_end_id-batch_start_id
    batch_features = np.zeros((this_batch_size, max_edges,
                               padded_num_edge_features))
    target_shape = (this_batch_size, max_edges, targets[0].shape[1]) if (
        all_targets) else (this_batch_size, max_edges)
    batch_targets = np.ones(target_shape) * nan_coding_value
    for i in range(this_batch_size):
      data_id = shuffled_data_ids[batch_start_id+i]
      molecule_features = features[data_id]
      molecule_targets = targets[data_id]
      num_edges = molecule_features.shape[0]
      
      batch_features[i, :num_edges, :num_edge_features] = molecule_features
      valid_target_rows = np.where(~np.isnan(molecule_targets[:, 0]))[0]
      if all_targets:
        batch_targets[i, :molecule_targets.shape[0]] = molecule_targets
      else:
        batch_targets[i, valid_target_rows] = molecule_targets[
            valid_target_rows, 0]
        
    if auxiliary_losses:
      batch_targets = np.repeat(batch_targets[:, :, np.newaxis],
                                transformer_depth, axis=2)
    
    yield batch_features, batch_targets
    
    batch_in_epoch = (batch_in_epoch + 1) % num_batches_per_epoch
  
  
def get_model(hyperpars):
  if hyperpars['model_save_name'] == 'initial_transformer':
    return models.initial_transformer
  
  raise ValueError('Model save name not supported')
  
  
def log_mae(y, p, s):
  valid_sc = np.arange(8)
  log_mae_sc = np.zeros_like(valid_sc)
  for i in range(valid_sc.size):
    sc = valid_sc[i]
    log_mae_sc = np.log(np.abs(y[s==sc]-p[s==sc]).mean())
  
  return log_mae_sc.mean()


#load_all_graphs('test', recompute_graphs=True)
#load_all_graphs('train', recompute_graphs=True)
#load_all_graphs('train', target_graph=True, recompute_graphs=True)
