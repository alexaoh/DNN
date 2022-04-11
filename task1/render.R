# File to render Rmd into both html and pdf.

my.path <- "."
document.name <- "task1"
fn <- file.path(my.path, document.name)
rmarkdown::render(paste0(fn, ".Rmd"), 
                  output_format =  c("html_document","pdf_document"),
                  output_file = c(paste0(fn, output_format ='.html'),
                                  paste0(fn, output_format ='.pdf')),
                  params = list(seed = 111),
                  encoding = "UTF8")
