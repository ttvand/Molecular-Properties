import math
import numpy as np
import transformer_train_validate
import utils


# Main parameters
mode = ['train', 'validate', 'train_validate'][2]

hyperpars = {
    'seed': 14,
    
    # Model parameters
    'model_save_name': 'initial_transformer',
    'transformer_num_heads': 8,
    'transformer_residual_dropout': 0.1,
    'transformer_attention_dropout': 0.1,
    'transformer_depth': 6,
    'prediction_layers': [],
    'auxiliary_losses': True,
    
    # Data / training parameters.
    'batch_size_train': 32,
    'batch_size_valid_train': 256,
    'batch_size_valid': 1024,
    'validation_fraction': 0.2,
    'initial_lr': 1e-3,
    'epochs': 100,
    'nan_coding_value': 999,
    
    'override_saved_model': True,
}


# Load the train graphs if they have not been loaded before
if (not 'TRAIN_GRAPHS' in locals()) and (not 'TRAIN_GRAPHS' in globals()):
  TRAIN_TARGET_GRAPHS, _, _ = utils.load_all_graphs(
      'train', graph_nx_format=False, only_edge=True, target_graph=True)
  TRAIN_GRAPHS, _, _ = utils.load_all_graphs(
      'train', graph_nx_format=False, only_edge=True)

# Add the maximum number of edges and number of features to the hyperparameters
max_edges = np.array([g.shape[0] for g in TRAIN_GRAPHS]).max()
hyperpars['max_edges'] = max_edges
hyperpars['num_edge_features'] = TRAIN_GRAPHS[0].shape[1]
hyperpars['padded_num_edge_features'] = math.ceil(
    TRAIN_GRAPHS[0].shape[1]/hyperpars['transformer_num_heads'])*(
        hyperpars['transformer_num_heads'])

# Determine the train and validation ids with a fixed seed for reproducible
# results.
num_graphs = len(TRAIN_GRAPHS)
np.random.seed(hyperpars['seed'])
permuted_ids = np.random.permutation(num_graphs)
train_ids = permuted_ids[:int(num_graphs*(1-hyperpars['validation_fraction']))]
valid_ids = np.setdiff1d(np.arange(num_graphs), train_ids)
model_save_path = '../Models/' + hyperpars['model_save_name'] + '.h5'

# Train the model
if 'train' in mode:
  transformer_train_validate.train(
      hyperpars, TRAIN_GRAPHS, TRAIN_TARGET_GRAPHS, train_ids, valid_ids,
      model_save_path)

# Evaluate the model predictions
if 'validate' in mode:
  transformer_train_validate.validate(
      hyperpars, TRAIN_GRAPHS, TRAIN_TARGET_GRAPHS, valid_ids, model_save_path)