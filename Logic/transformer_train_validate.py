import itertools
import math
import numpy as np
import os
import utils
import warnings


from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


# Main train function for unevenly spaced time series data
def train(hyperpars, features, targets, train_ids, valid_ids, save_path):
  K.clear_session()
  if hyperpars['override_saved_model'] or not os.path.exists(save_path):
    train_gen = utils.generator_batch(
        features, targets, train_ids, hyperpars, hyperpars['batch_size_train'])
    validation_gen = utils.generator_batch(
        features, targets, valid_ids, hyperpars,
        hyperpars['batch_size_valid_train'])
    train_model = utils.get_model(hyperpars)
    inputs, outputs = train_model(hyperpars)
    model = Model(inputs, outputs)
    adam = Adam(lr=hyperpars['initial_lr'])
    model.compile(optimizer=adam, loss=utils.make_masked_mae(hyperpars))
    (monitor, monitor_mode) = ('val_loss', 'min')
    plot_model(model, to_file=save_path[:-3] + '.png', show_shapes=True)
    checkpointer = ModelCheckpoint(save_path, monitor=monitor,
                                   mode=monitor_mode, verbose=1,
                                   save_best_only=True, save_freq='epoch')
    callbacks = [checkpointer]
    callbacks.append(utils.PlotLosses())
#    callbacks.append(TensorBoard('./Graph'))
    
    model.fit_generator(
        train_gen,
        steps_per_epoch=math.ceil(train_ids.size/hyperpars['batch_size_train']),
        epochs=hyperpars['epochs'],
        callbacks=callbacks,
        validation_data=validation_gen,
        validation_steps=math.ceil(valid_ids.size/hyperpars[
            'batch_size_valid_train']),
        verbose=1,
        )
  
  
# Validate CPC model by computing the predict ratio of positive CPC pairs to
# the negative CPC pairs
def validate(hyperpars, features, targets, valid_ids, save_path):
  validation_gen = utils.generator_batch(
        features, targets, valid_ids, hyperpars,
        hyperpars['batch_size_valid'], all_targets=True)
  
  # Generate the validation data by calling the generator *N* times
  num_valid_batches = math.ceil(valid_ids.size/hyperpars['batch_size_valid'])
  valid_data = list(itertools.islice(validation_gen, num_valid_batches))
  valid_preds = make_predictions(save_path, valid_data)
  
  valid_targets = np.concatenate([d[1] for d in valid_data])
  import pdb; pdb.set_trace()
  log_mae = utils.log_mae(valid_targets[:, :, 0], valid_preds,
                          valid_targets[:, :, 5])
  print('Log mean absolute error: {0:.5f}'.format(log_mae))
  
  return log_mae
  
# Helper function for generating model predictions
def make_predictions(save_path, data, model=None):
  if model is None:
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)
      model = load_model(save_path, custom_objects={}, compile=False)
  model_inputs = []
  model_inputs = np.concatenate([d[0] for d in data])
  preds = model.predict(model_inputs, verbose=1)
  
  return preds