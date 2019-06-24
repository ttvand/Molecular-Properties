# rm(list=ls())
library(data.table)

mode <- c("train", "test")[1]
reload_data_sources <- TRUE
reload_data_sources <- TRUE

data_folder <- "/media/tom/cbd_drive/Kaggle/Molecular Properties/Data/"

# Sources
# Bonds and charges: https://www.kaggle.com/asauve/dataset-with-number-of-bonds-between-atoms

# Lukyanenko brute force feature engineering: https://www.kaggle.com/artgor/brute-force-feature-engineering 
# LightGBM features: https://www.kaggle.com/kabure/lightgbm-full-pipeline-model

# Coulomb forces: https://www.kaggle.com/kernels/scriptcontent/15751144/notebook
# https://www.kaggle.com/rio114/coulomb-interaction-speed-up
# https://www.kaggle.com/brandenkmurray/coulomb-interaction-parallelized
# https://www.kaggle.com/kernels/scriptcontent/15751144/notebook

# Karplus equation for 3J coupling constants: https://www.kaggle.com/c/champs-scalar-coupling/discussion/93793#latest-548247
# Angle computation: https://www.kaggle.com/kmat2019/effective-feature
# https://en.wikipedia.org/wiki/Dihedral_angle
if(reload_data_sources || !exists("original_edges")){
  original_edges <- fread(file.path(data_folder, paste0(mode, ".csv")))
  original_edges[, id:=NULL]
  structures <- fread(file.path(data_folder, "structures.csv"))
  
  # Make sure that atom_index_0 < atom_index_1
  min_edge_index <- pmin(original_edges$atom_index_0,
                         original_edges$atom_index_1)
  max_edge_index <- pmax(original_edges$atom_index_0,
                         original_edges$atom_index_1)
  original_edges$atom_index_0 <- min_edge_index
  original_edges$atom_index_1 <- max_edge_index
  
  if (mode == "train"){
    scalar_coupling_contributions <- fread(
      file.path(data_folder, "scalar_coupling_contributions.csv"))
    original_edges <- cbind(original_edges,
                            scalar_coupling_contributions[, 5:8, with=FALSE])
  }
  
  molecules <- unique(original_edges$molecule_name)
  original_edges <- original_edges[molecule_name %in% molecules, ]
  original_nodes <- structures[molecule_name %in% molecules, ]
}

edges <- original_edges
nodes <- original_nodes

#########################
# Add bonds and charges #
#########################
bonds <- fread(file.path(data_folder, paste0(mode, "_bonds.csv")))
charges <- fread(file.path(data_folder, paste0(mode, "_charges.csv")))
bonds[, V1:=NULL]
charges[, V1:=NULL]
nodes <- merge(nodes, charges, by=c("molecule_name", "atom_index"))
edge_primary_keys <- c("molecule_name", "atom_index_0", "atom_index_1")
edges <- merge(edges, bonds, by=edge_primary_keys, all=TRUE)
edges[is.na(nbond), nbond:=0]
edges[is.na(error), error:=0]
edges[!is.na(type), bond_type:=type]
edges$bond_type_0 <- as.numeric(gsub("[^0-9\\.]", "", edges$bond_type))

# Add L2 distance for non direct bonds 2/3JH(C/H/N)
node_distances <- merge(original_nodes, original_nodes, by="molecule_name",
                        allow.cartesian=TRUE, suffixes = c("_0", "_1"))
node_distances <- node_distances[atom_index_0 != atom_index_1]
node_distances[, distance:=sqrt((x_0-x_1)**2 + (y_0-y_1)**2 + (z_0-z_1)**2)]
node_distances_xyz1 <- node_distances[, c("x_1", "y_1", "z_1")]
node_distances[, c("x_0", "y_0", "z_0", "x_1", "y_1", "z_1") := NULL]
edges <- merge(edges, node_distances, by=edge_primary_keys)
edges$L2dist <- edges$distance # It was verified that these agree!
edges[, distance:=NULL]

#######################################
# Add Lukyanenko brute force features #
#######################################
additional_edge <- cbind(copy(node_distances), node_distances_xyz1)

# Drop rows that don't occur in the edge data to make sure that the aggregate
# distance measures are only based on relevant edges
# Note: distances are also based on bonds that have no target values
edge_basic <- edges[, c("molecule_name", "atom_index_0", "atom_index_1",
                        "bond_type", "bond_type_0")]
edge_basic_flipped <- copy(edge_basic)
edge_basic_flipped$atom_index_1 <- edge_basic$atom_index_0
edge_basic_flipped$atom_index_0 <- edge_basic$atom_index_1
additional_edge <- merge(additional_edge, rbind(edge_basic, edge_basic_flipped),
                         by=edge_primary_keys)

additional_edge[, `:=`(molecule_dist_mean=mean(distance),
                       molecule_dist_min=min(distance),
                       molecule_dist_max=max(distance)), .(molecule_name)]

additional_edge[, `:=`(
  atom_0_couples_count=.N,
  atom_0_couples_count_direct=sum(bond_type_0==1),
  # molecule_atom_index_0_x_1_std=sd(x_1),
  # molecule_atom_index_0_y_1_mean=mean(y_1),
  # molecule_atom_index_0_y_1_std=sd(y_1),
  # molecule_atom_index_0_y_1_max=max(y_1),
  # molecule_atom_index_0_z_1_std=sd(z_1),
  molecule_atom_index_0_dist_mean=mean(distance),
  molecule_atom_index_0_dist_max=max(distance),
  molecule_atom_index_0_dist_min=min(distance),
  molecule_atom_index_0_dist_std=sd(distance)),
  .(molecule_name, atom_index_0)]
additional_edge[atom_0_couples_count==1, `:=`(
  # molecule_atom_index_0_x_1_std=0,
  # molecule_atom_index_0_y_1_std=0,
  # molecule_atom_index_0_z_1_std=0,
  molecule_atom_index_0_dist_std=0)
  ]

additional_edge[, `:=`(
  # molecule_atom_index_0_y_1_mean_diff=molecule_atom_index_0_y_1_mean-y_1,
  # molecule_atom_index_0_y_1_max_diff=molecule_atom_index_0_y_1_max-y_1,
  molecule_atom_index_0_dist_mean_diff=molecule_atom_index_0_dist_mean-distance,
  molecule_atom_index_0_dist_mean_div=molecule_atom_index_0_dist_mean/distance,
  molecule_atom_index_0_dist_max_diff=molecule_atom_index_0_dist_max-distance,
  molecule_atom_index_0_dist_max_div=molecule_atom_index_0_dist_max/distance,
  molecule_atom_index_0_dist_min_diff=molecule_atom_index_0_dist_min-distance,
  molecule_atom_index_0_dist_min_div=molecule_atom_index_0_dist_min/distance,
  molecule_atom_index_0_dist_std_diff=molecule_atom_index_0_dist_std-distance,
  molecule_atom_index_0_dist_std_div=molecule_atom_index_0_dist_std/distance)
  ]


additional_edge[, `:=`(
  atom_1_couples_count=.N,
  atom_1_couples_count_direct=sum(bond_type_0==1),
  molecule_atom_index_1_dist_mean=mean(distance),
  molecule_atom_index_1_dist_max=max(distance),
  molecule_atom_index_1_dist_min=min(distance),
  molecule_atom_index_1_dist_std=sd(distance)),
  .(molecule_name, atom_index_1)]
additional_edge[atom_1_couples_count==1, `:=`(
  molecule_atom_index_1_dist_std=0)
  ]

additional_edge[, `:=`(
  molecule_atom_index_1_dist_mean_diff=molecule_atom_index_1_dist_mean-distance,
  molecule_atom_index_1_dist_mean_div=molecule_atom_index_1_dist_mean/distance,
  molecule_atom_index_1_dist_max_diff=molecule_atom_index_1_dist_max-distance,
  molecule_atom_index_1_dist_max_div=molecule_atom_index_1_dist_max/distance,
  molecule_atom_index_1_dist_min_diff=molecule_atom_index_1_dist_min-distance,
  molecule_atom_index_1_dist_min_div=molecule_atom_index_1_dist_min/distance,
  molecule_atom_index_1_dist_std_diff=molecule_atom_index_1_dist_std-distance,
  molecule_atom_index_1_dist_std_div=molecule_atom_index_1_dist_std/distance)
  ]


additional_edge[, `:=`(
  molecule_atom_0_dist_mean=mean(distance),
  molecule_atom_0_dist_min=min(distance),
  molecule_atom_0_dist_std=sd(distance)),
  .(molecule_name, atom_0)]
additional_edge[atom_0_couples_count==1, `:=`(
  molecule_atom_0_dist_std=0)
  ]

additional_edge[, `:=`(
  molecule_atom_0_dist_min_diff=molecule_atom_0_dist_min-distance,
  molecule_atom_0_dist_min_div=molecule_atom_0_dist_min/distance,
  molecule_atom_0_dist_std_diff=molecule_atom_0_dist_std-distance)
  ]


additional_edge[, `:=`(
  molecule_atom_1_dist_mean=mean(distance),
  molecule_atom_1_dist_min=min(distance),
  molecule_atom_1_dist_std=sd(distance)),
  .(molecule_name, atom_1)]
additional_edge[atom_1_couples_count==1, `:=`(
  molecule_atom_1_dist_std=0)
  ]

additional_edge[, `:=`(
  molecule_atom_1_dist_min_diff=molecule_atom_1_dist_min-distance,
  molecule_atom_1_dist_min_div=molecule_atom_1_dist_min/distance,
  molecule_atom_1_dist_std_diff=molecule_atom_1_dist_std-distance)
  ]


additional_edge[, `:=`(
  molecule_type_0_dist_std=sd(distance)),
  .(molecule_name, bond_type_0)]

additional_edge[, `:=`(
  molecule_type_0_dist_std_diff=molecule_type_0_dist_std-distance)
  ]


additional_edge[, `:=`(
  molecule_type_dist_mean=mean(distance),
  molecule_type_dist_max=max(distance),
  molecule_type_dist_min=min(distance),
  molecule_type_dist_std=sd(distance)),
  .(molecule_name, bond_type)]

additional_edge[, `:=`(
  molecule_type_dist_mean_diff=molecule_type_dist_mean-distance,
  molecule_type_dist_mean_div=molecule_type_dist_mean/distance,
  molecule_type_dist_std_diff=molecule_type_dist_std-distance)
  ]

# Join with nodes and edges
additional_edge[, c("atom_0", "bond_type", "bond_type_0",
                    "atom_1", "x_1", "y_1", "z_1") := NULL]
edges <- merge(edges, additional_edge, by=edge_primary_keys)
edge_unique_columns <- c()
top_additional_edge <- additional_edge[1:1000]
for(col in tail(colnames(additional_edge), -3)){
  summ <- top_additional_edge[
    , .(num_unique=length(unique(eval(as.name(col))))), molecule_name]
  if(min(summ$num_unique)==1){
    edge_unique_columns <- c(edge_unique_columns, col)
  }
}
node_columns <- intersect(
  edge_unique_columns, c("atom_0_couples_count",
                         "atom_0_couples_count_direct",
                         "molecule_atom_index_0_y_1_mean",
                         "molecule_atom_index_0_y_1_std",
                         "molecule_atom_index_0_y_1_max",
                         "molecule_atom_index_0_z_1_std",
                         "molecule_atom_index_0_y_1_mean_diff",
                         "molecule_atom_index_0_y_1_max_diff"))

nodes <- merge(
  nodes, additional_edge[
    rowidv(additional_edge, cols=c("molecule_name", "atom_index_0"))==1,
    c(edge_primary_keys[1:2], node_columns), with=FALSE],
  by.x=c("molecule_name", "atom_index"), by.y=edge_primary_keys[1:2])


##################
# Coulomb forces #
##################
coulomb <- fread(file.path(data_folder,
                           paste0(mode, "_inv_squared_distances.csv")))
nodes <- merge(nodes, coulomb, by=c("molecule_name", "atom_index"))


##################
# Angle features #
##################
angles <- fread(file.path(data_folder, paste0(mode, "_angles.csv")))
edges <- merge(edges, angles, by=c(edge_primary_keys, "type"), all=TRUE)
edges[is.na(distance_0), distance_0:=L2dist]
edges[is.na(distance_1), distance_1:=L2dist]
edges[is.na(cos_0_1), cos_0_1:=0]
edges[is.na(cos_0), cos_0:=0]
edges[is.na(cos_1), cos_1:=0]
edges[is.na(interm_0_atom), interm_0_atom:='None']
edges[is.na(interm_1_atom), interm_1_atom:='None']
edges[is.na(atom_0_interm_0_interm_1_cosine),
      atom_0_interm_0_interm_1_cosine:=0]
edges[is.na(interm_0_interm_1_atom_1_cosine),
      interm_0_interm_1_atom_1_cosine:=0]
edges[is.na(dihedral_angle), dihedral_angle:=0]
edges[is.na(`2Jdirect_cosine`), `2Jdirect_cosine`:=0]
edges[, dihedral_cos:=cos(dihedral_angle)]
edges[, dihedral_sin:=sin(dihedral_angle)]
edges[, dihedral_angle:=NULL]


###########################
# Compute global features #
###########################
global_cols_dist <- intersect(
  edge_unique_columns, c("molecule_dist_mean",
                         "molecule_dist_min",
                         "molecule_dist_max"))
globals <- additional_edge[
  rowidv(additional_edge, cols=c("molecule_name"))==1,
  c(edge_primary_keys[1], global_cols_dist), with=FALSE]

global_node_counts <- nodes[, .(
  num_nodes = .N, num_H = sum(atom=="H"), num_C = sum(atom=="C"),
  num_N = sum(atom=="N"), num_O = sum(atom=="O"), num_F = sum(atom=="F")),
  molecule_name]
global_node_counts[, `:=` (
  H_frac = num_H/num_nodes, C_frac = num_C/num_nodes, N_frac = num_N/num_nodes,
  O_frac = num_O/num_nodes, F_frac = num_F/num_nodes)]
globals <- global_node_counts[globals]
global_edge_counts <- edges[, .(num_edges=.N), molecule_name]
globals <- global_edge_counts[globals]


########################################
# Store node, edge and global features #
########################################
# Verify that there are no NA values before writing
stopifnot(sum(sapply(edges, function(x) sum(is.na(x))) > 0) %in% c(1, 6))
stopifnot(max(sapply(nodes, function(x) sum(is.na(x)))) == 0)
stopifnot(max(sapply(globals, function(x) sum(is.na(x)))) == 0)
fwrite(edges, file.path(data_folder, paste0(mode, "_extended_edges.csv")))
fwrite(nodes, file.path(data_folder, paste0(mode, "_extended_nodes.csv")))
fwrite(globals, file.path(data_folder, paste0(mode, "_extended_globals.csv")))
