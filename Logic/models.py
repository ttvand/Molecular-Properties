#import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
#from tensorflow.keras.layers import Reshape

#from tensorflow.keras.models import Sequential

from keras_transformer.transformer import TransformerBlock

def initial_transformer(hyperpars):
  # Define the dimensions of the inputs
  max_edges = hyperpars['max_edges']
  num_edge_features = hyperpars['padded_num_edge_features']
  auxiliary_losses = hyperpars['auxiliary_losses']
  prediction_layers = hyperpars['prediction_layers']
  
  inputs = Input((max_edges, num_edge_features), name='inputs')
  
  x = inputs
  auxiliary_outputs = []
  for step in range(hyperpars['transformer_depth']):
    transformer_block = TransformerBlock(
    name='transformer' + str(step+1),
    num_heads=hyperpars['transformer_num_heads'],
    residual_dropout=hyperpars['transformer_residual_dropout'],
    attention_dropout=hyperpars['transformer_attention_dropout'],
    use_masking=False,
    vanilla_wiring=False,
    )
    x = transformer_block(x)
    if auxiliary_losses:
      y = x
      for layer in prediction_layers + [1]:
        y = Dense(layer)(y)
      auxiliary_outputs.append(y)
  
  if auxiliary_losses:
    x = Lambda(lambda x: K.concatenate(x, axis=2))(auxiliary_outputs)
  else:
    for layer in prediction_layers + [1]:
      x = Dense(layer)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
  outputs = x
  
  return (inputs, outputs)
