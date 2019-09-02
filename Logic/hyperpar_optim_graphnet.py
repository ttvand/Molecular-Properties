from collections import OrderedDict
import datetime
import numpy as np
import os
import pandas as pd
import random
import utils


num_random_hyperpars = 1000
model_save_name = 'initial_graphnet'
sweep_save_folder = '/home/tom/Kaggle/Molecular-Properties/Data/Hyper Sweeps/'
validation_fraction = 0.2
continuing_experiment = ['19-06-21-17-12.csv', None][1]

fixed_params = {
    'seed': 14,
    'num_training_iterations': 1e9,
    'max_train_seconds': 3600*12,
    'batch_size_train': 32,
    'batch_size_valid_train': 256,
    'batch_size_valid': 1024,
    'edge_output_layer_norm': False,
    
    'log_every_seconds': 120,
    }

# In order of apparent effect on OOF performance
param_grid = {
    'num_processing_steps': [8, 16], # (message-passing) steps.
    'learning_rate': [1e-3],
    'inverse_relative_error_weights': [False, True],
    'skip_connection_encoder_decoder': [False, True],
    'separate_edge_output': [False, True],
    'latent_size': [128, 256],
    'num_layers': [4],
}

# Load the train graphs if they have not been loaded before
if (not 'TRAIN_GRAPHS' in locals()) and (not 'TRAIN_GRAPHS' in globals()):
  TRAIN_GRAPHS, EDGE_PERMUTATIONS, molecule_names = utils.load_all_graphs(
      'train')
  TRAIN_TARGET_GRAPHS, _, _ = utils.load_all_graphs('train', target_graph=True)

# Determine the train and validation ids.
num_graphs = len(TRAIN_GRAPHS)
np.random.seed(fixed_params['seed'])
permuted_ids = np.random.permutation(num_graphs)
train_ids = permuted_ids[:int(num_graphs*(1-validation_fraction))]
valid_ids = np.setdiff1d(np.arange(num_graphs), train_ids)
model_save_path_base = '../Models/' + model_save_name

the_date = datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
base_sweep_path = sweep_save_folder + 'hyperpar_sweep_initial_graphnet '
sweep_path = base_sweep_path + the_date + '.csv'
sweep_summaries = []
# Optionally, continue an existing experiment
if continuing_experiment is not None:
  continuing_path = base_sweep_path + continuing_experiment
  if os.path.exists(continuing_path):
    sweep_path = continuing_path
    continuing_data = pd.read_csv(continuing_path)
    data_cols = continuing_data.columns
    sweep_summaries = []
    for i in range(continuing_data.shape[0]):
      data_row = continuing_data.iloc[i]
      value_tuple = [(c, data_row[c]) for c in data_cols]
      sweep_summaries.append(OrderedDict(value_tuple))

for i in range(num_random_hyperpars):
  print('Random hyperpar setting {} of {}'.format(i+1, num_random_hyperpars))
  model_save_path = model_save_path_base + '_' + str(i) + '.ckpt'
  hyperpars = {k: random.choice(v) for k, v in param_grid.items()}
  
  num_processing_steps = 8 if i % 2 == 0 else 16
  hyperpars['num_processing_steps'] = num_processing_steps
  separate_edge_output = i % 4 > 2
  hyperpars['separate_edge_output'] = separate_edge_output
  
  selected_grid = OrderedDict(sorted(hyperpars.items()))
  hyperpars.update(fixed_params)
  utils.train_model(hyperpars, TRAIN_GRAPHS, TRAIN_TARGET_GRAPHS,
                    EDGE_PERMUTATIONS, train_ids, valid_ids, model_save_path)
  valid_score, valid_mae = utils.validate_model(
      hyperpars, TRAIN_GRAPHS, TRAIN_TARGET_GRAPHS, EDGE_PERMUTATIONS,
      valid_ids, model_save_path)
  validation = OrderedDict()
  validation['Score'] = valid_score
  validation['MAE'] = valid_mae
  summary_dict = OrderedDict(list(validation.items()) + list(selected_grid.items()))
  sweep_summaries.append(summary_dict)
  
  sweep_results = pd.DataFrame(sweep_summaries)
  sweep_results.to_csv(sweep_path, index=False)
