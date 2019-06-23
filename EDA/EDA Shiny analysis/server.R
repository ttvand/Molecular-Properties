shinyServer(function(input, output, session) {
  # Decrement the data start period 
  observeEvent(input$prev_molecule, {
    updateSliderInput(session, "molecule_id",
                      value=max(c(1, input$molecule_id-1)))
  })
  
  # Increment the data start period 
  observeEvent(input$next_molecule, {
    updateSliderInput(session, "molecule_id",
                      value=min(c(num_molecules, input$molecule_id+1)))
  })
  
  # Plot the raw quake training time series data
  output$molecule_plotly <- renderPlotly({
    molecule_id <- input$molecule_id
    
    # Plot a certain molecule by first plotting the edges and then the connections
    plot_molecule <- molecules[molecule_id]
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
    
    p
  })
})