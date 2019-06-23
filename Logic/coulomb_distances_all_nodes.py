# Adapted from https://www.kaggle.com/tvdwiele/speed-up-coulomb-interaction-56x-faster-47696a/edit
import pandas as pd
import numpy as np
import datetime as dt

data_folder = '/media/tom/cbd_drive/Kaggle/Molecular Properties/Data/'
#data_folder = '/home/tom/Kaggle/Molecular Properties/Data/'
source = ['train', 'test'][1]

df_structures = pd.read_csv(data_folder + 'structures.csv')
df_train = pd.read_csv(data_folder + source + '.csv')

df_distance = df_structures.merge(
    df_structures, how = 'left', on= 'molecule_name', suffixes = ('_0', '_1'))
df_distance = df_distance.loc[
    df_distance['atom_index_0'] != df_distance['atom_index_1']]
df_distance['distance'] = np.linalg.norm(
    df_distance[['x_0','y_0', 'z_0']].values - 
    df_distance[['x_1', 'y_1', 'z_1']].values, axis=1, ord = 2)

def get_interaction_data_frame(df_distance, num_nearest=5):
  time_start = dt.datetime.now()
  print("START", time_start)
  
  # get nearest 5 (num_nearest) by distances
  print("Starting distance grouping")
  df_distance = df_distance.drop(['atom_0', 'x_0', 'y_0', 'z_0',
                                  'atom_index_1', 'x_1', 'y_1', 'z_1'], 1)
  df_temp = df_distance.groupby(['molecule_name', 'atom_index_0', 'atom_1'])[
      'distance']
  df_temp = df_temp.nsmallest(num_nearest)
  
  # make it clean
  df_temp = pd.DataFrame(df_temp).reset_index()[[
      'molecule_name', 'atom_index_0', 'atom_1', 'distance']]
  df_temp.columns = ['molecule_name', 'atom_index', 'atom', 'distance']
  
  time_nearest = dt.datetime.now()
  print("Time Nearest", time_nearest-time_start)
  
  # get rank by distance
  df_temp['distance_rank'] = df_temp.groupby([
      'molecule_name', 'atom_index', 'atom'])['distance'].rank(
  ascending = True, method = 'first').astype(int)
  
  time_rank = dt.datetime.now()
  print("Time Rank", time_rank-time_nearest)
  
  # pivot to get nearest distance by atom type 
  df_distance_nearest = pd.pivot_table(
      df_temp, index = ['molecule_name','atom_index'],
      columns= ['atom', 'distance_rank'], values= 'distance')
  
  time_pivot = dt.datetime.now()
  print("Time Pivot", time_pivot-time_rank)
  del df_temp
  
  columns_distance_nearest =  np.core.defchararray.add(
      'distance_nearest_',  np.array(
          df_distance_nearest.columns.get_level_values(
              'distance_rank')).astype(str) + np.array(
              df_distance_nearest.columns.get_level_values('atom')))
  df_distance_nearest.columns = columns_distance_nearest
  
  # 1 / r^2 to get the square inverse same with the previous kernel
  df_distance_sq_inv_farthest = 1 / (df_distance_nearest ** 2)
  
  columns_distance_sq_inv_farthest = [col.replace(
      'distance_nearest', 'distance_sq_inv_farthest') for col in columns_distance_nearest]

  df_distance_sq_inv_farthest.columns = columns_distance_sq_inv_farthest
  time_inverse = dt.datetime.now()
  print("Time Inverse Calculation", time_inverse-time_pivot)
  
  df_interaction = pd.concat([df_distance_sq_inv_farthest, df_distance_nearest] , axis = 1)
  df_interaction.reset_index(inplace = True)
  
  time_concat = dt.datetime.now()
  print("Time Concat", time_concat-time_inverse)
  
  return df_interaction
  
considered_molecules = np.intersect1d(
    df_train['molecule_name'].unique(),
    df_structures['molecule_name'].unique()#[:650]
    )

interaction_list = []
batch_size = 100
num_batches = max(1, considered_molecules.size // batch_size)
for i in range(num_batches):
  print("\nProcessing batch {} of {}".format(i+1, num_batches))
  if i == num_batches-1:
    considered_molecules_batch = considered_molecules[i*batch_size:]
  else:
    considered_molecules_batch = considered_molecules[
        i*batch_size:(i+1)*batch_size]
  df_interaction_batch = get_interaction_data_frame(df_distance.loc[
      df_distance['molecule_name'].isin(considered_molecules_batch)])
  interaction_list.append(df_interaction_batch)
df_interaction = pd.concat(interaction_list)
df_interaction = df_interaction.fillna(0)

df_interaction.to_csv(data_folder + source + '_inv_squared_distances.csv',
                      index=False)

