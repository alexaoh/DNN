


if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()


library(keras)
K <- keras::backend()

library(magick)
library(viridis)


model <- application_vgg16(weights = "imagenet")


img_path <- "/Users/frc/Documents/Docencia/ub/curs21_22/dl/airplanes_2.jpg"
img <- image_load(img_path, target_size = c(224, 224)) %>%
  image_to_array() %>%
  array_reshape(dim = c(1, 224, 224, 3)) %>%
  imagenet_preprocess_input()



preds <- model %>% predict(img)
imagenet_decode_predictions(preds, top = 3)[[1]]
which.max(preds[1,])


object_output <- model$output[, which.max(preds[1,])]
last_conv_layer <- model %>% get_layer("block5_conv3")
grads <- K$gradients(object_output, last_conv_layer$output)[[1]]
pooled_grads <- K$mean(grads, axis = c(0L, 1L, 2L))
iterate <- K$`function`(list(model$input),
                        list(pooled_grads, last_conv_layer$output[1,,,]))
c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))
for (i in 1:512) {
  conv_layer_output_value[,,i] <-
    conv_layer_output_value[,,i] * pooled_grads_value[[i]]
}
heatmap <- apply(conv_layer_output_value, c(1,2), mean)



heatmap <- pmax(heatmap, 0)
heatmap <- heatmap / max(heatmap)
write_heatmap <- function(heatmap, filename, width = 224, height = 224,
                          bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}
write_heatmap(heatmap, "airpllanes_heatmap_2.png")




image <- image_read(img_path)
info <- image_info(image)
geometry <- sprintf("%dx%d!", info$width, info$height)
pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal)))
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
write_heatmap(heatmap, "airplanes_overlay_2.png",
              width = 32, height = 32, bg = NA, col = pal_col)
image_read("airplanes_overlay_2.png") %>%
  image_resize(geometry, filter = "quadratic") %>%
  image_composite(image, operator = "blend", compose_args = "20") %>%
  plot()





