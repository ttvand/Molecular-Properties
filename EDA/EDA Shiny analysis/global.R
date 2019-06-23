# Load required libraries
library(shiny)
library(shinythemes)
library(data.table)
library(plotly)

data_folder <- "/media/tom/cbd_drive/Kaggle/Molecular Properties/Data/"

if (!exists('train') || !exists('structures')){
  train <- fread(file.path(data_folder, "train.csv"))
  structures <- fread(file.path(data_folder, "structures.csv"))
  molecules <- unique(train$molecule_name)
  num_molecules <- length(molecules)
}

edge_colors <- list(H="grey", C="black", N="blue", O="red", F="green")
edge_sizes <- list(H=4, C=6, N=7, O=8, F=9)

init_rand_id <- 487#sample(1:num_molecules, 1)

