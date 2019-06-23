library(data.table)
library(plotly)

# Much better visualization: https://www.kaggle.com/borisdee/how-to-easy-visualization-of-molecules
data_folder <- "/media/tom/cbd_drive/Kaggle/Molecular Properties/Data/"

if (!exists('train') || !exists('structures')){
  train <- fread(file.path(data_folder, "train.csv"))
  structures <- fread(file.path(data_folder, "structures.csv"))
  molecules <- unique(train$molecule_name)
}

edge_colors = list(H="grey", C="black", N="blue", O="red", F="green")
edge_sizes = list(H=4, C=6, N=7, O=8, F=9)

# Plot a certain molecule by first plotting the edges and then the connections
plot_molecule <- molecules[6000]
nodes <- structures[molecule_name == plot_molecule]
edges <- train[molecule_name == plot_molecule]

nodes$color <- sapply(nodes$atom, function(x) edge_colors[[x]])
nodes$size <- sapply(nodes$atom, function(x) edge_sizes[[x]])
nodes$atom <- as.factor(nodes$atom)
ax <- list(title="", zeroline=FALSE, showline=FALSE, showticklabels=FALSE,
           showgrid=!TRUE)
p <- plot_ly(nodes, x=~x, y=~y, z=~z, color=~atom,
             colors=unlist(edge_colors)) %>%
  add_markers(size=~size) %>%
  layout(scene=list(xaxis=ax, yaxis=ax, zaxis=ax))

# Now add the direct connections

print(p)
