# Shiny app for Birdapp main file. 
#setwd("/home/ajo/gitRepos/DNN/birdapp")

library(shiny)
library(shinydashboard)
library(rsconnect)
library(keras)
library(tensorflow)
library(tidyverse)

# Could make a button in the app to switch between the two models!
# Could be interesting to do this, for fun!

#mod <- load_model_hdf5("www/finetunedXception1.h5") # This is the initial model. 
mod <- load_model_hdf5("www/tunedHypParamXception.h5")
load("www/label_list.RData")
target.size <- c(224,224,3)
options(scipen=999) # Prevent scientific number formatting.

# Define the ui object. 
ui <- dashboardPage(
  skin = "black",
  title = "BirdApp - Classify your Bird!",
  
  #(1) Header
  
  dashboardHeader(title=tags$h1("BirdApp - Classify your Bird!",style="font-size: 120%; font-weight: bold; color: black"),
                  titleWidth = 350,
                  tags$li(class = "dropdown"),
                  dropdownMenu(type = "notifications", icon = icon("question-circle", "fa-1x"), badgeStatus = NULL,
                               headerText="",
                               tags$li("Created by alexaoh, but"), 
                               tags$li(a(href = "https://forloopsandpiepkicks.wordpress.com",
                                         target = "_blank",
                                         tagAppendAttributes(icon("exclamation-circle"), class = "info"),
                                         "(heavily) inspired by (click here)"))
                               
                  )),
  
  
  #(2) Sidebar
  
  dashboardSidebar(
    width=350,
    fileInput("input_image","File" ,accept = c('.jpg','.jpeg')), 
    tags$br(),
    tags$p("Upload the image here (We only accept jpg and jpeg formats at this point).")
  ),
  
  
  #(3) Body
  
  dashboardBody(
    
    h4("Instruction:"),
    tags$br(),tags$p("1. Take a picture of a bird."),
    tags$p("2. Crop image so that bird fills out most of the image."),
    tags$p("3. Upload image with menu on the left."),
    tags$br(),
    
    fluidRow(
      column(h4("Image:"),imageOutput("output_image"), width=6),
      column(h4("Result:"),tags$br(),textOutput("warntext",), tags$br(),
             tags$p("This bird is probably a:"),tableOutput("text"),width=6)
    ),tags$br()
    
  ))


# We create the server object. 
server <- function(input, output) {
  
  image <- reactive({image_load(input$input_image$datapath, target_size = target.size[1:2])})
  
  prediction <- reactive({
    if(is.null(input$input_image)){return(NULL)}
    x <- image_to_array(image())
    x <- array_reshape(x, c(1, dim(x)))
    x <- x/255
    pred <- mod %>% predict(x)
    pred <- data.frame("Bird" = label.list, "Prediction" = t(pred))
    pred <- pred[order(pred$Prediction, decreasing=T),][1:5,]
    pred$Prediction <- sprintf("%.2f %%", 100*pred$Prediction)
    pred
  })
  
  output$text <- renderTable({
    prediction()
  })
  
  output$warntext <- renderText({
    req(input$input_image)
    
    if(as.numeric(substr(prediction()[1,2],1,4)) >= 45){return(NULL)}
    warntext <- "Warning: I am not sure about this bird!"
    warntext
  })
  
  
  output$output_image <- renderImage({
    req(input$input_image)
    
    outfile <- input$input_image$datapath
    contentType <- input$input_image$type
    list(src = outfile,
         contentType=contentType,
         width = 400)
  }, deleteFile = TRUE)
  
}

shinyApp(ui, server)
#rsconnect::deployApp()
