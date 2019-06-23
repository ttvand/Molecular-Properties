rm(list=ls())
library(data.table)

data_folder <- "/media/tom/cbd_drive/Kaggle/Molecular Properties/Data/"

train <- fread(file.path(data_folder, "train.csv"))
test <- fread(file.path(data_folder, "test.csv"))
structures <- fread(file.path(data_folder, "structures.csv"))

# Inspect if the fraction of coupling types is the same in train and test - OK!
train_summ <- train[, list(.N, max_index=max(atom_index_0, atom_index_1)),
                    molecule_name]
train_summ_type <- train[, .(
  .N,
  mean_abs_target=mean(abs(scalar_coupling_constant)),
  min_target=min(scalar_coupling_constant),
  max_target=max(scalar_coupling_constant)), type]
train_summ_type <- train_summ_type[order(type)]
test_summ_type <- test[, .N, type]
test_summ_type <- test_summ_type[order(type)]
train_summ_type$N/test_summ_type$N

