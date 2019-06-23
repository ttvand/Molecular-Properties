# Define UI
shinyUI(navbarPage(
  "Exploratory data analysis",
  theme = shinytheme("cerulean"), #united
  tabPanel("Train molecule visualization",
           sidebarPanel(
             sliderInput("molecule_id", "Molecule ID", 1, num_molecules,
                         init_rand_id, 1),
             fluidRow(
               column(2, actionButton("prev_molecule", "",
                                      icon("arrow-alt-circle-left"))),
               column(2, actionButton("next_molecule", "",
                                      icon("arrow-alt-circle-right")))
             ),
             width = 3),
           mainPanel(
             br(),
             plotlyOutput("molecule_plotly")
           )
  )
))

