
library(keras)
library(tidyverse)
library(data.table)
library(raster)


fashion_mnist <- dataset_fashion_mnist()
x_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
x_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y


# Checking first few clothes

showCloth <- function(data, label) {

    image(
       flip(raster(data %>% unlist()),1),
       col = gray.colors(255), xlab = label, ylab = ""
   )
}


showFirst6Cloth <- function(data, label) {
    par(mfrow = c(2, 3))
    if (missing(label)) label <- rep("", 6)
    walk(1:6, ~showCloth(data[.,,], label[.]))
}

showFirst6Cloth(x_train,y_train)
showFirst6Cloth(x_test)

# 
# showFirst6Cloth(x_train[1,,], y_train[1:6])
# 
# 
# x_train[1:6,,]


# taking only first 500 pics for training

# Let's create a validation set from the training set and not use the test set 
x_train_modeling <- x_train[1:500,,]
y_train_modeling <- y_train[1:500] %>% to_categorical(10)

x_valid_modeling <- x_train[501:1000,,]
y_valid_modeling <- y_train[501:1000] %>%  to_categorical(10)

x_test_modeling <- x_test[1:1000,,]
y_test_modeling <- y_test[1:1000] %>%  to_categorical(10)

# normalising data to useable  format for modeling
x_train_modeling <- rbindlist(
    lapply(1:nrow(x_train_modeling),
           function(y) {lapply(1:28, function(x) {
               x_train_modeling[y,x,] 
           }) %>% unlist %>% as.list
    })
) %>%  as.data.table()

x_valid_modeling <- rbindlist(
    lapply(1:nrow(x_valid_modeling),
           function(y) {lapply(1:28, function(x) {
               x_valid_modeling[y,x,] 
           }) %>% unlist %>% as.list
           })
) %>% as.data.table()

x_test_modeling <- rbindlist(
    lapply(1:nrow(x_test_modeling),
           function(y) {lapply(1:28, function(x) {
               x_test_modeling[y,x,] 
           }) %>% unlist %>% as.list
           })
) %>% as.data.table()

colnames(x_train_modeling) <- paste0('pixel_', 1:784)
colnames(x_valid_modeling) <- paste0('pixel_', 1:784)
colnames(x_test_modeling) <- paste0('pixel_', 1:784)


# do further transformations for x vars 

x_train_modeling <- as.matrix(x_train_modeling) / 255
x_valid_modeling <- as.matrix(x_valid_modeling) / 255
x_test_modeling <- as.matrix(x_test_modeling) / 255


# training a neural net

model <- keras_model_sequential()
model %>%
    layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 10, activation = 'softmax')

summary(model)

# modify the model in place
model %>% 
    compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
)

# fit the model

history <- model %>%  fit(
    x_train_modeling, y_train_modeling,
    epochs = 30, batch_size = 128,
    validation_data = list(x_valid_modeling, y_valid_modeling)
)

plot(history) + theme_bw()

# evaluate model 

pred_1 <- model %>% evaluate(x_test_modeling, y_test_modeling)
keras_predictions_valid <- predict_classes(model, x_valid_modeling)


plotConfusionMatrix <- function(label, prediction) {
    bind_cols(label = label, predicted = prediction) %>%
        group_by(label, predicted) %>%
        summarize(N = n()) %>%
        ggplot(aes(label, predicted)) +
        geom_tile(aes(fill = N), colour = "white") +
        scale_x_continuous(breaks = 0:9) +
        scale_y_continuous(breaks = 0:9) +
        geom_text(aes(label = N), vjust = 1, color = "white") +
        scale_fill_viridis_c() +
        theme_bw() + theme(legend.position = "none")
}

plotConfusionMatrix(y_train[501:1000], keras_predictions_valid)

# experiment with different layers

# Experiment 1 
model2 <- keras_model_sequential()


model2 %>%
    layer_dense(units = 150, activation = 'linear', input_shape = c(784)) %>%
    layer_dropout(rate = 0.25) %>% 
    layer_dense(units = 10, activation = 'softmax')

model2 %>% compile(
    optimizer = 'adam', 
    loss = 'categorical_crossentropy',
    metrics = c('accuracy')
)

model2 %>%  fit(
    x_train_modeling, y_train_modeling,
    epochs = 30, batch_size = 128,
    validation_data = list(x_valid_modeling, y_valid_modeling)
)

pred_2 <- model2 %>% evaluate(x_test_modeling, y_test_modeling)

# Experiment 2 

model3 <- keras_model_sequential()

model3 %>%
    layer_dense(units = 150, activation = 'relu', input_shape = c(784)) %>%
    layer_dropout(rate = 0.25) %>% 
    layer_dense(units = 30, activation = 'tanh') %>% 
    layer_dense(units = 10, activation = 'softmax')

model3 %>% compile(
    optimizer = optimizer_rmsprop(), 
    loss = 'categorical_crossentropy',
    metrics = c('accuracy')
)

model3 %>%  fit(
    x_train_modeling, y_train_modeling,
    epochs = 50, batch_size = 128,
    validation_data = list(x_valid_modeling, y_valid_modeling)
)

pred_3 <- model3 %>% evaluate(x_test_modeling, y_test_modeling)


# Experiment 3

model4 <- keras_model_sequential()

model4 %>%
    layer_dense(units = 150, activation = 'relu', input_shape = c(784)) %>%
    layer_dropout(rate = 0.25) %>% 
    layer_dense(units = 30, activation = 'relu') %>% 
    layer_dense(units = 10, activation = 'softmax')

model4 %>% compile(
    optimizer = optimizer_rmsprop(), 
    loss = 'categorical_crossentropy',
    metrics = c('accuracy')
)

model4 %>%  fit(
    x_train_modeling, y_train_modeling,
    epochs = 30, batch_size = 128,
    validation_data = list(x_valid_modeling, y_valid_modeling)
)

pred_4 <- model4 %>% evaluate(x_test_modeling, y_test_modeling)

# Experiment 4

model5 <- keras_model_sequential()

model5 %>%
    layer_dense(units = 150, activation = 'relu', input_shape = c(784)) %>%
    layer_dropout(rate = 0.3) %>% 
    layer_dense(units = 100, activation = 'tanh') %>% 
    layer_dense(units = 10, activation = 'softmax')

model5 %>% compile(
    optimizer = optimizer_rmsprop(), 
    loss = 'categorical_crossentropy',
    metrics = c('accuracy')
)

model5 %>%  fit(
    x_train_modeling, y_train_modeling,
    epochs = 60, batch_size = 128,
    validation_data = list(x_valid_modeling, y_valid_modeling)
)

pred_5 <- model5 %>% evaluate(x_test_modeling, y_test_modeling)

# Experiment 5

model6 <- keras_model_sequential()

model6 %>%
    layer_dense(units = 150, activation = 'relu', input_shape = c(784)) %>%
    layer_dropout(rate = 0.3) %>% 
    layer_dense(units = 100, activation = 'tanh') %>%
    layer_dense(units = 50, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')

model6 %>% compile(
    optimizer = optimizer_rmsprop(), 
    loss = 'categorical_crossentropy',
    metrics = c('accuracy')
)

model6 %>%  fit(
    x_train_modeling, y_train_modeling,
    epochs = 60, batch_size = 128,
    validation_data = list(x_valid_modeling, y_valid_modeling),
    callbacks = list(
        callback_reduce_lr_on_plateau(monitor = 'val_loss', factor = 0.1)
    )
)

pred_6 <- model6 %>% evaluate(x_test_modeling, y_test_modeling)


# Convolutional networks --------------------------------------------------

x_train_cnn <- array_reshape(x_train_modeling, c(nrow(x_train_modeling), 28, 28, 1))
x_valid_cnn <- array_reshape(x_valid_modeling, c(nrow(x_valid_modeling), 28, 28, 1))
x_test_cnn <- array_reshape(x_test_modeling, c(nrow(x_test_modeling), 28, 28, 1))

cnn_model <- keras_model_sequential()
cnn_model %>%
    layer_conv_2d(
        filters = 32,
        kernel_size = c(3, 3),
        activation = 'relu',
        input_shape = c(28, 28, 1)
    ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.25) %>%
    layer_flatten() %>%
    layer_dense(units = 16, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')

compile(
    cnn_model,
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
)

fit(
    cnn_model, x_train_cnn, y_train_modeling,
    epochs = 100, batch_size = 128,
    validation_data = list(x_valid_cnn, y_valid_modeling)#,
 #   callbacks = list(
  #      callback_reduce_lr_on_plateau(monitor = 'val_loss', factor = 0.1)
  #  )
)

pred_cnn <- cnn_model %>% evaluate(x_test_cnn, y_test_modeling)


# Experiment 2 

cnn_model2 <- keras_model_sequential()
cnn_model2 %>%
    layer_conv_2d(
        filters = 32,
        kernel_size = c(3, 3),
        activation = 'relu',
        input_shape = c(28, 28, 1)
    ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.25) %>%
    layer_flatten() %>%
    layer_dense(units = 132, activation = 'relu') %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 10, activation = 'softmax')

compile(
    cnn_model2,
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
)

fit(
    cnn_model2, x_train_cnn, y_train_modeling,
    epochs = 100, batch_size = 128,
    validation_data = list(x_valid_cnn, y_valid_modeling)#,
    #   callbacks = list(
    #      callback_reduce_lr_on_plateau(monitor = 'val_loss', factor = 0.1)
    #  )
)

pred_cnn2 <- cnn_model2 %>% evaluate(x_test_cnn, y_test_modeling)


# Experiment 3 

cnn_model3 <- keras_model_sequential()
cnn_model3 %>%
    layer_conv_2d(
        filters = 32,
        kernel_size = c(3, 3),
        activation = 'relu',
        input_shape = c(28, 28, 1)
    ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.25) %>%
    layer_flatten() %>%
    layer_dense(units = 132, activation = 'relu') %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 50, activation = 'tanh') %>% 
    layer_dropout(rate = 0.1) %>% 
    layer_dense(units = 30, activation = 'relu') %>% 
    layer_dense(units = 10, activation = 'softmax')

compile(
    cnn_model3,
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
)

fit(
    cnn_model3, x_train_cnn, y_train_modeling,
    epochs = 60, batch_size = 128,
    validation_data = list(x_valid_cnn, y_valid_modeling)#,
    #   callbacks = list(
    #      callback_reduce_lr_on_plateau(monitor = 'val_loss', factor = 0.1)
    #  )
)

pred_cnn3 <- cnn_model3 %>% evaluate(x_test_cnn, y_test_modeling)


# Experiment 4
cnn_model4 <- keras_model_sequential()
cnn_model4 %>%
    layer_conv_2d(
        filters = 64,
        kernel_size = c(3, 3),
        activation = 'relu',
        input_shape = c(28, 28, 1)
    ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.25) %>%
    layer_flatten() %>%
    layer_dense(units = 150, activation = 'relu') %>%
    layer_dropout(rate = 0.25) %>%
    layer_dense(units = 30, activation = 'tanh') %>% 
    layer_dense(units = 10, activation = 'softmax')

compile(
    cnn_model4,
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
)

fit(
    cnn_model4, x_train_cnn, y_train_modeling,
    epochs = 60, batch_size = 128,
    validation_data = list(x_valid_cnn, y_valid_modeling)#,
    #   callbacks = list(
    #      callback_reduce_lr_on_plateau(monitor = 'val_loss', factor = 0.1)
    #  )
)

pred_cnn4 <- cnn_model4 %>% evaluate(x_test_cnn, y_test_modeling)
