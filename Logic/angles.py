# Adapted from https://www.kaggle.com/kmat2019/effective-feature
import multiprocessing as mp
import numpy as np
import pandas as pd


data_folder = '/media/tom/cbd_drive/Kaggle/Molecular Properties/Data/'
#data_folder = '/home/tom/Kaggle/Molecular Properties/Data/'

source = ['train', 'test'][1]
recompute_features = False
parallel = True


def map_atom_info(df_1, df_2, atom_idx):
  df = pd.merge(df_1, df_2, how = 'left',
                left_on  = ['molecule_name', 'atom_index_' + str(atom_idx)],
                right_on = ['molecule_name',  'atom_index'])
  df = df.drop('atom_index', axis=1)

  return df


def make_features(df):
    df['dx']=df['x_1']-df['x_0']
    df['dy']=df['y_1']-df['y_0']
    df['dz']=df['z_1']-df['z_0']
    df['distance']=(df['dx']**2+df['dy']**2+df['dz']**2)**(1/2)
    return df
  
  
def add_cos_features(df):
  df["distance_0"]=((df['x_0']-df['x_closest_0'])**2+(df['y_0']-df['y_closest_0'])**2+(df['z_0']-df['z_closest_0'])**2)**(1/2)
  df["distance_1"]=((df['x_1']-df['x_closest_1'])**2+(df['y_1']-df['y_closest_1'])**2+(df['z_1']-df['z_closest_1'])**2)**(1/2)
  df["vec_0_x"]=(df['x_0']-df['x_closest_0'])/df["distance_0"]
  df["vec_0_y"]=(df['y_0']-df['y_closest_0'])/df["distance_0"]
  df["vec_0_z"]=(df['z_0']-df['z_closest_0'])/df["distance_0"]
  df["vec_1_x"]=(df['x_1']-df['x_closest_1'])/df["distance_1"]
  df["vec_1_y"]=(df['y_1']-df['y_closest_1'])/df["distance_1"]
  df["vec_1_z"]=(df['z_1']-df['z_closest_1'])/df["distance_1"]
  df["vec_x"]=(df['x_1']-df['x_0'])/df["distance"]
  df["vec_y"]=(df['y_1']-df['y_0'])/df["distance"]
  df["vec_z"]=(df['z_1']-df['z_0'])/df["distance"]
  df["cos_0_1"]=df["vec_0_x"]*df["vec_1_x"]+df["vec_0_y"]*df["vec_1_y"]+df["vec_0_z"]*df["vec_1_z"]
  df["cos_0"]=df["vec_0_x"]*df["vec_x"]+df["vec_0_y"]*df["vec_y"]+df["vec_0_z"]*df["vec_z"]
  df["cos_1"]=df["vec_1_x"]*df["vec_x"]+df["vec_1_y"]*df["vec_y"]+df["vec_1_z"]*df["vec_z"]
  df=df.drop(['vec_0_x','vec_0_y','vec_0_z','vec_1_x','vec_1_y','vec_1_z','vec_x','vec_y','vec_z'], axis=1)
  return df


# Adapted from https://stackoverflow.com/questions/11285613/selecting-multiple-columns-in-a-pandas-dataframe
def cosine_angle(nodes, node_ids):
  a = nodes.iloc[node_ids[0]][['x', 'y', 'z']].values.astype(float)
  b = nodes.iloc[node_ids[1]][['x', 'y', 'z']].values.astype(float)
  c = nodes.iloc[node_ids[2]][['x', 'y', 'z']].values.astype(float)
  
  ba = a - b
  bc = c - b
  
  cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
  return cosine_angle


# Adapted from https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
def dihedral_angle(nodes, node_ids):
  p0 = nodes.iloc[node_ids[0]][['x', 'y', 'z']].values.astype(float)
  p1 = nodes.iloc[node_ids[1]][['x', 'y', 'z']].values.astype(float)
  p2 = nodes.iloc[node_ids[2]][['x', 'y', 'z']].values.astype(float)
  p3 = nodes.iloc[node_ids[3]][['x', 'y', 'z']].values.astype(float)

  b0 = -1.0*(p1 - p0)
  b1 = p2 - p1
  b2 = p3 - p2

  b0xb1 = np.cross(b0, b1)
  b1xb2 = np.cross(b2, b1)

  b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

  y = np.dot(b0xb1_x_b1xb2, b1)*(1.0/np.linalg.norm(b1))
  x = np.dot(b0xb1, b1xb2)

  return np.arctan2(y, x)
  

def add_angle_features(edges, bonds, nodes):
  # 2J and 3J connections
  num_edges = edges.shape[0]
  num_bonds = bonds.shape[0]
  edges.type = edges.type_x
  for r in range(num_edges):
    edge_length = edges.iloc[r].type_x[0]
    if edge_length in ['2', '3']:
      max_trajectory_length = int(edge_length)
      bonds_0_values = bonds.atom_index_0.values
      bonds_1_values = bonds.atom_index_1.values
      all_row_valid_ids = np.ones((num_bonds), dtype=bool)
      valid_trajectories = [([int(edges.iloc[r].atom_index_0)],
                             all_row_valid_ids)]
      considered_id = 0
      
      while len(valid_trajectories) > considered_id:
        if len(valid_trajectories[0][0]) < (max_trajectory_length+1):
          sequence, valid_ids = valid_trajectories[considered_id]
          del valid_trajectories[considered_id]
          
          if len(sequence) == max_trajectory_length:
            valid_ids[np.logical_and(
                bonds_0_values != edges.iloc[r].atom_index_1,
                bonds_1_values != edges.iloc[r].atom_index_1)] = False
          
          valid_rows_0 = np.where(np.logical_and(
              valid_ids, bonds_0_values == sequence[-1]))[0]
          valid_rows_1 = np.where(np.logical_and(
              valid_ids, bonds_1_values == sequence[-1]))[0]
            
          for row in valid_rows_0:
            updated_valid = valid_ids
            updated_valid[row] = False
            valid_trajectories.append((sequence + [bonds_1_values[row]],
                                       updated_valid))
                                       
          for row in valid_rows_1:
            updated_valid = valid_ids
            updated_valid[row] = False
            valid_trajectories.append((sequence + [bonds_0_values[row]],
                                       updated_valid))
        else:
          considered_id = considered_id + 1 
       
      if len(valid_trajectories) != 1:
        # Likely cause: incorrect inferred bond structure - check for updates
        # to the bond script later at: https://www.kaggle.com/asauve/dataset-with-number-of-bonds-between-atoms
        print("ERROR: {} instead of 1 possible trajectory! {} {} {} {}".format(
            len(valid_trajectories), edges.molecule_name.values[0],
            edges.type.values[r], edges.atom_index_0.values[r],
            edges.atom_index_1.values[r]))
      else:
        edge_nodes = valid_trajectories[0][0]
        row_index = edges.index[int(r)]
        # Add feature for 2J connections
        if int(edge_length) == 2:
          
          edges.at[row_index, '2Jdirect_cosine'] = cosine_angle(nodes, edge_nodes)
        
        # Add features for 3J connections
        elif int(edge_length) == 3:
          edges.at[row_index, 'interm_0_atom'] = nodes.iloc[
              edge_nodes[1]].atom
          edges.at[row_index, 'interm_1_atom'] = nodes.iloc[
              edge_nodes[2]].atom
          edges.at[row_index, 'atom_0_interm_0_interm_1_cosine'] = cosine_angle(
              nodes, edge_nodes[:-1])
          edges.at[row_index, 'interm_0_interm_1_atom_1_cosine'] = cosine_angle(
              nodes, edge_nodes[1:])
          edges.at[row_index, 'dihedral_angle'] = dihedral_angle(
              nodes, edge_nodes)
          
#        df_edges['interm_0_atom'] = 'None'
#        df_edges['interm_1_atom'] = 'None'
#        df_edges['atom_0_interm_0_interm_1_cosine'] = -2
#        df_edges['interm_0_interm_1_atom_1_cosine'] = -2
#        df_edges['dihedral_cosine'] = -2
#        df_edges['2Jdirect_cosine'] = -2
    
  edges = edges.drop(['id', 'scalar_coupling_constant', 'atom_0',
                      'x_0', 'y_0', 'z_0', 'atom_1', 'x_1', 'y_1', 'z_1', 'dx',
                      'dy', 'dz', 'distance', 'type_x', 'atom_index_closest_0',
                      'distance_closest_0', 'type_y', 'x_closest_0',
                      'y_closest_0', 'z_closest_0', 'atom_index_closest_1',
                      'distance_closest_1', 'x_closest_1', 'y_closest_1',
                      'z_closest_1'], axis=1)
  return edges


def get_next_ids(df, name, start_id, considered_nexts = 10000):
  sub_df = df[start_id:(start_id+considered_nexts)]
  match_ids = np.where(sub_df.molecule_name == name)[0]
  assert(match_ids.size > 0 and match_ids[-1] < (considered_nexts-1))
  
  return start_id + match_ids[-1] + 1, start_id + match_ids


if recompute_features or not 'df_nodes' in locals():
  df_nodes = pd.read_csv(data_folder + 'structures.csv')
  df_bonds = pd.read_csv(data_folder + source + '_bonds.csv')
  df_edges = pd.read_csv(data_folder + source + '.csv')
  
  # Make sure that atom_index_0 < atom_index_1
  min_edge_index = np.minimum(df_edges.atom_index_0.values,
                              df_edges.atom_index_1.values)
  max_edge_index = np.maximum(df_edges.atom_index_0.values,
                              df_edges.atom_index_1.values)
  df_edges.atom_index_0 = min_edge_index
  df_edges.atom_index_1 = max_edge_index
  
  for atom_idx in [0, 1]:
    df_edges = map_atom_info(df_edges, df_nodes, atom_idx)
    df_edges = df_edges.rename(columns={'atom': 'atom_' + str(atom_idx),
                                        'x': 'x_' + str(atom_idx),
                                        'y': 'y_' + str(atom_idx),
                                        'z': 'z_' + str(atom_idx)})
    
  df_edges=make_features(df_edges)
  
  #I apologize for my poor coding skill. Please make the better one.
  df_temp=df_edges.loc[:,["molecule_name","atom_index_0","atom_index_1","type",
                          "distance","x_0","y_0","z_0","x_1","y_1","z_1"]].copy()
  
  df_temp_=df_temp.copy()
  df_temp_= df_temp_.rename(columns={'atom_index_0': 'atom_index_1',
                                     'atom_index_1': 'atom_index_0',
                                     'x_0': 'x_1',
                                     'y_0': 'y_1',
                                     'z_0': 'z_1',
                                     'x_1': 'x_0',
                                     'y_1': 'y_0',
                                     'z_1': 'z_0'})
  df_temp=pd.concat((df_temp,df_temp_),axis=0)
  
  df_temp["min_distance"]=df_temp.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('min')
  df_temp= df_temp[df_temp["min_distance"]==df_temp["distance"]]
  
  df_temp=df_temp.drop(['x_0','y_0','z_0','min_distance'], axis=1)
  df_temp= df_temp.rename(columns={'atom_index_0': 'atom_index',
                                   'atom_index_1': 'atom_index_closest',
                                   'distance': 'distance_closest',
                                   'x_1': 'x_closest',
                                   'y_1': 'y_closest',
                                   'z_1': 'z_closest'})
  
  # Delete duplicated rows (some atom pairs have perfectly same distance)
  # This code is added based on Adriano Avelar's comment.
  df_temp=df_temp.drop_duplicates(subset=['molecule_name', 'atom_index'])
  
  for atom_idx in [0,1]:
    df_edges = map_atom_info(df_edges, df_temp, atom_idx)
    df_edges = df_edges.rename(columns={'atom_index_closest': 'atom_index_closest_' + str(atom_idx),
                                        'distance_closest': 'distance_closest_' + str(atom_idx),
                                        'x_closest': 'x_closest_' + str(atom_idx),
                                        'y_closest': 'y_closest_' + str(atom_idx),
                                        'z_closest': 'z_closest_' + str(atom_idx)})
  df_edges=add_cos_features(df_edges)

# Loop over all molecules and add angle features to df_edges
molecule_names = np.unique(df_bonds.molecule_name.values)

# Note: These are the problematic molecules which do not contain 2- or 3- way
# bonds as expected - Most probably due to incorrect bond calculations
molecule_names_defect = [
    'dsgdb9nsd_004015', 'dsgdb9nsd_007375', 'dsgdb9nsd_007654',
    'dsgdb9nsd_013833', 'dsgdb9nsd_014318', 'dsgdb9nsd_017808',
    'dsgdb9nsd_018308', 'dsgdb9nsd_022116', 'dsgdb9nsd_022195',
    'dsgdb9nsd_022203', 'dsgdb9nsd_022232', 'dsgdb9nsd_022238',
    'dsgdb9nsd_022460', 'dsgdb9nsd_022504', 'dsgdb9nsd_022686',
    'dsgdb9nsd_023150', 'dsgdb9nsd_023372', 'dsgdb9nsd_027272',
    'dsgdb9nsd_028251', 'dsgdb9nsd_029540', 'dsgdb9nsd_032465',
    'dsgdb9nsd_033400', 'dsgdb9nsd_042424', 'dsgdb9nsd_043213',
    'dsgdb9nsd_045541', 'dsgdb9nsd_045546', 'dsgdb9nsd_046611',
    'dsgdb9nsd_049723', 'dsgdb9nsd_050309', 'dsgdb9nsd_050620',
    'dsgdb9nsd_050736', 'dsgdb9nsd_053821', 'dsgdb9nsd_054102',
    'dsgdb9nsd_054124', 'dsgdb9nsd_054409', 'dsgdb9nsd_054412',
    'dsgdb9nsd_054569', 'dsgdb9nsd_054611', 'dsgdb9nsd_054629',
    'dsgdb9nsd_054691', 'dsgdb9nsd_054692', 'dsgdb9nsd_054794',
    'dsgdb9nsd_054796', 'dsgdb9nsd_054994', 'dsgdb9nsd_055187',
    'dsgdb9nsd_055190', 'dsgdb9nsd_055410', 'dsgdb9nsd_055450',
    'dsgdb9nsd_055476', 'dsgdb9nsd_055477', 'dsgdb9nsd_055479',
    'dsgdb9nsd_055701', 'dsgdb9nsd_055944', 'dsgdb9nsd_058281',
    'dsgdb9nsd_058982', 'dsgdb9nsd_059827', 'dsgdb9nsd_059977',
    'dsgdb9nsd_066499', 'dsgdb9nsd_066506', 'dsgdb9nsd_074137',
    'dsgdb9nsd_074176', 'dsgdb9nsd_081049', 'dsgdb9nsd_081054',
    'dsgdb9nsd_081057', 'dsgdb9nsd_081567', 'dsgdb9nsd_081568',
    'dsgdb9nsd_081581', 'dsgdb9nsd_083401', 'dsgdb9nsd_083414',
    'dsgdb9nsd_083417', 'dsgdb9nsd_084310', 'dsgdb9nsd_086636',
    'dsgdb9nsd_089622', 'dsgdb9nsd_090696', 'dsgdb9nsd_091519',
    'dsgdb9nsd_093347', 'dsgdb9nsd_093567', 'dsgdb9nsd_093572',
    'dsgdb9nsd_093986', 'dsgdb9nsd_093988', 'dsgdb9nsd_094182',
    'dsgdb9nsd_095438', 'dsgdb9nsd_099728', 'dsgdb9nsd_100092',
    'dsgdb9nsd_100734', 'dsgdb9nsd_102015', 'dsgdb9nsd_108410',
    'dsgdb9nsd_112990', 'dsgdb9nsd_113174', 'dsgdb9nsd_113176',
    'dsgdb9nsd_117643', 'dsgdb9nsd_118448', 'dsgdb9nsd_121600',
    'dsgdb9nsd_121780', 'dsgdb9nsd_121882', 'dsgdb9nsd_122767',
    'dsgdb9nsd_128418', 'dsgdb9nsd_130336', 'dsgdb9nsd_131200',
    
]

#molecule_names = molecule_names_defect
#df_bonds = df_bonds[df_bonds.molecule_name.isin(molecule_names)]
#df_edges = df_edges[df_edges.molecule_name.isin(molecule_names)]
#df_nodes = df_nodes[df_nodes.molecule_name.isin(molecule_names)]

# Add columns with default values
df_edges['interm_0_atom'] = 'None'
df_edges['interm_1_atom'] = 'None'
df_edges['atom_0_interm_0_interm_1_cosine'] = 0.0
df_edges['interm_0_interm_1_atom_1_cosine'] = 0.0
df_edges['dihedral_angle'] = 0.0
df_edges['2Jdirect_cosine'] = 0.0
# Obtain all node, edge and global ids
print('Obtain all node, edge and global ids')
all_bonds = []
all_edges = []
all_nodes = []
start_id_bonds = 0
start_id_edges = 0
start_id_nodes = 0
for graph_id, name in enumerate(molecule_names):
  start_id_bonds, bond_ids = get_next_ids(df_bonds, name, start_id_bonds)
  start_id_edges, edge_ids = get_next_ids(df_edges, name, start_id_edges)
  start_id_nodes, node_ids = get_next_ids(df_nodes, name, start_id_nodes)
  all_bonds.append(df_bonds.iloc[bond_ids])
  all_edges.append(df_edges.iloc[edge_ids])
  all_nodes.append(df_nodes.iloc[node_ids])

angle_dfs = []
if parallel:
  pool = mp.Pool(processes=mp.cpu_count()-1)
  results = [pool.apply_async(
      add_angle_features, args=(
          e, b, n,)) for (e, b, n) in zip(all_edges, all_bonds, all_nodes)]
  angle_dfs = [p.get() for p in results]
else:
  for i, name in enumerate(molecule_names):
    print('Molecule {} of {}.'.format(i+1, len(molecule_names)))
    angle_dfs.append(add_angle_features(all_edges[i], all_bonds[i],
                                        all_nodes[i]))

# Concat angle df's and overwrite invalid angle features
angle_edges = pd.concat(angle_dfs, axis=0)
invalid_molecules_edge_ids = np.logical_or(
    np.logical_and(angle_edges['type'].apply(lambda x: x[0]).values == '2',
                   angle_edges['2Jdirect_cosine'] < -1.5),
    np.logical_and(angle_edges['type'].apply(lambda x: x[0]) == '3',
                   angle_edges.dihedral_angle == -2.0),
                   )
invalid_molecules_edge_names = np.unique(angle_edges[
    invalid_molecules_edge_ids].molecule_name)
invalid_edge_rows = angle_edges.molecule_name.isin(
    invalid_molecules_edge_names)
angle_edges.ix[invalid_edge_rows, ['interm_0_atom', 'interm_1_atom']] = 'None'
angle_edges.ix[invalid_edge_rows, [
    'atom_0_interm_0_interm_1_cosine', 'interm_0_interm_1_atom_1_cosine',
    '2Jdirect_cosine']] = 0.0
angle_edges.ix[invalid_edge_rows, ['dihedral_angle']] = 0.0
angle_edges.to_csv(data_folder + source + '_angles.csv', index=False)