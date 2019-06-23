# Adapted from https://www.kaggle.com/tvdwiele/coulomb-interaction-speed-up/edit
import pandas as pd
import numpy as np
import time

data_folder = '/media/tom/cbd_drive/Kaggle/Molecular Properties/Data/'
#data_folder = '/home/tom/Kaggle/Molecular Properties/Data/'
source = ['train', 'test'][0]

df_structures = pd.read_csv(data_folder + 'structures.csv')
df_train = pd.read_csv(data_folder + source + '.csv')

def get_dist_matrix(df_structures_idx, molecule):
  df_temp = df_structures_idx.loc[molecule]
  locs = df_temp[['x','y','z']].values
  num_atoms = len(locs)
  loc_tile = np.tile(locs.T, (num_atoms,1,1))
  dist_mat = ((loc_tile - loc_tile.T)**2).sum(axis=1)
  return dist_mat


def assign_atoms_index(df_idx, molecule):
  import pdb; pdb.set_trace()
  se_0 = df_idx.loc[molecule]['atom_index_0']
  se_1 = df_idx.loc[molecule]['atom_index_1']
  if type(se_0) == np.int64:
    se_0 = pd.Series(se_0)
  if type(se_1) == np.int64:
    se_1 = pd.Series(se_1)
  assign_idx = pd.concat([se_0, se_1]).unique()
  assign_idx.sort()
  return assign_idx


def get_pickup_dist_matrix(df_idx, df_structures_idx, molecule, num_pickup=5, atoms=['H', 'C', 'N', 'O', 'F']):
  pickup_dist_matrix = np.zeros([0, len(atoms)*num_pickup])
  assigned_idxs = assign_atoms_index(df_idx, molecule) # [0, 1, 2, 3, 4, 5, 6] -> [1, 2, 3, 4, 5, 6]
  dist_mat = get_dist_matrix(df_structures_idx, molecule)
  for idx in assigned_idxs: # [1, 2, 3, 4, 5, 6] -> [2]
    df_temp = df_structures_idx.loc[molecule]
    locs = df_temp[['x','y','z']].values

    dist_arr = dist_mat[idx] # (7, 7) -> (7, )

    atoms_mole = df_structures_idx.loc[molecule]['atom'].values # ['O', 'C', 'C', 'N', 'H', 'H', 'H']
    atoms_mole_idx = df_structures_idx.loc[molecule]['atom_index'].values # [0, 1, 2, 3, 4, 5, 6]

    mask_atoms_mole_idx = atoms_mole_idx != idx # [ True,  True, False,  True,  True,  True,  True]
    masked_atoms = atoms_mole[mask_atoms_mole_idx] # ['O', 'C', 'N', 'H', 'H', 'H']
    masked_atoms_idx = atoms_mole_idx[mask_atoms_mole_idx]  # [0, 1, 3, 4, 5, 6]
    masked_dist_arr = dist_arr[mask_atoms_mole_idx]  # [ 5.48387003, 2.15181049, 1.33269675, 10.0578779, 4.34733927, 4.34727838]
    masked_locs = locs[masked_atoms_idx]

    sorting_idx = np.argsort(masked_dist_arr) # [2, 1, 5, 4, 0, 3]
    sorted_atoms_idx = masked_atoms_idx[sorting_idx] # [3, 1, 6, 5, 0, 4]
    sorted_atoms = masked_atoms[sorting_idx] # ['N', 'C', 'H', 'H', 'O', 'H']
    sorted_dist_arr = 1/masked_dist_arr[sorting_idx] #[0.75035825,0.46472494,0.23002898,0.23002576,0.18235297,0.09942455]

    target_matrix = np.zeros([len(atoms), num_pickup])
    for a, atom in enumerate(atoms):
      pickup_atom = sorted_atoms == atom # [False, False,  True,  True, False,  True]
      pickup_dist = sorted_dist_arr[pickup_atom] # [0.23002898, 0.23002576, 0.09942455]

      num_atom = len(pickup_dist)
      if num_atom > num_pickup:
        target_matrix[a, :num_pickup] = pickup_dist[:num_pickup]
      else:
        target_matrix[a, :num_atom] = pickup_dist
    
    pickup_dist_matrix = np.vstack([pickup_dist_matrix, target_matrix.reshape(-1)])
  return pickup_dist_matrix #(num_atoms, num_pickup*5)
  
df_structures_idx = df_structures.set_index('molecule_name')
import pdb; pdb.set_trace()
df_train_idx = df_train.set_index('molecule_name')


num = 5
mols = df_train['molecule_name'].unique()
num_div = len(mols) // 5
dist_mat = np.zeros([0, num*5])
atoms_idx = np.zeros([0], dtype=np.int32)
molecule_names = np.empty([0])

start = time.time()

mol_name_arr_list = []
assigned_idxs_list = []
dist_mat_mole_list = []
for i, mol in enumerate(mols[:3]):
  print("Processing molecule {} of {}".format(i+1, mols.size))
  assigned_idxs = assign_atoms_index(df_train_idx, mol)
  dist_mat_mole = get_pickup_dist_matrix(df_train_idx, df_structures_idx, mol, num_pickup=num)
  mol_name_arr = [mol] * len(assigned_idxs)
  mol_name_arr_list.append(mol_name_arr)
  assigned_idxs_list.append(assigned_idxs)
  dist_mat_mole_list.append(dist_mat_mole)
  
molecule_names = np.hstack(mol_name_arr_list)
atoms_idx = np.hstack(assigned_idxs_list)
dist_mat = np.vstack(dist_mat_mole_list)
    
col_name_list = []
atoms = ['H', 'C', 'N', 'O', 'F']
for a in atoms:
  for n in range(num):
    col_name_list.append('dist_{}_{}'.format(a, n))
        
se_mole = pd.Series(molecule_names, name='molecule_name')
se_atom_idx = pd.Series(atoms_idx, name='atom_index')
df_dist = pd.DataFrame(dist_mat, columns=col_name_list)
df_distance = pd.concat([se_mole, se_atom_idx, df_dist], axis=1)
df_distance.to_csv(data_folder + source + '_inv_squared_distances.csv',
                   index=False)

elapsed_time = time.time() - start
print ("elapsed_time: {0}".format(elapsed_time) + " s")