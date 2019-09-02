import numpy as np
import utils


# Main parameters
mode = ['train', 'validate', 'train_validate'][0]
model_save_name = 'initial_graphnet'

hyperpars = {
  'seed': 14,
  
  # Model parameters
  'num_processing_steps': 16, # (message-passing) steps.
  'skip_connection_encoder_decoder': False,
  'edge_output_layer_norm': False,
  'separate_edge_output': True,
  'latent_size': 128,
  'num_layers': 4,
  
  # Data / training parameters.
  'num_training_iterations': 1e9,
  'max_train_seconds': 15*3600,
  
  'batch_size_train': 32,
  'batch_size_valid_train': 256,
  'batch_size_valid': 1024,
  'validation_fraction': 0.2,
  'learning_rate': 1e-3,
  'inverse_relative_error_weights': True,
  
  # How much time between logging and printing the current results.
  # Make sure that this is a lot larger than the time to save the model!
  'log_every_seconds': 120,
}

# Load the train graphs if they have not been loaded before
if (not 'TRAIN_GRAPHS' in locals()) and (not 'TRAIN_GRAPHS' in globals()):
  TRAIN_GRAPHS, EDGE_PERMUTATIONS, molecule_names = utils.load_all_graphs(
      'train')
  TRAIN_TARGET_GRAPHS, _, _ = utils.load_all_graphs('train', target_graph=True)

# Determine the train and validation ids.
num_graphs = len(TRAIN_GRAPHS)
np.random.seed(hyperpars['seed'])
permuted_ids = np.random.permutation(num_graphs)
train_ids = permuted_ids[:int(num_graphs*(1-hyperpars['validation_fraction']))]
valid_ids = np.setdiff1d(np.arange(num_graphs), train_ids)
model_save_path = '../Models/' + model_save_name + '.ckpt'

# Train the model
if 'train' in mode:
  utils.train_model(hyperpars, TRAIN_GRAPHS, TRAIN_TARGET_GRAPHS,
                    EDGE_PERMUTATIONS, train_ids, valid_ids, model_save_path)

# Evaluate the model predictions
if 'validate' in mode:
  utils.validate_model(hyperpars, TRAIN_GRAPHS, TRAIN_TARGET_GRAPHS,
                       EDGE_PERMUTATIONS, valid_ids, model_save_path)