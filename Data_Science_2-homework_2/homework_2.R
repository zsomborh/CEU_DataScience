rm(list = ls())

library(keras)
library(grid)
library(magick)
library(filesstrings)
root_folder <- ('C:/Workspace/R_directory/CEU/machine-learning-course/data/hot-dog-not-hot-dog')


# create valid folder
#dir.create(paste0(root_folder,'/validation'))
#dir.create(paste0(root_folder, '/validation/hot_dog'))
#dir.create(paste0(root_folder, '/validation/not_hot_dog')) 

# pick valid images randomly 

#move_hotdog<- sample(list.files(paste0(root_folder,'/train/hot_dog')),50)
#move_not_hotdog<- sample(list.files(paste0(root_folder,'/train/not_hot_dog')),50)

# move files to validation folder
#for (hotdog in move_hotdog){
#    file.move(paste0(root_folder,'/train/hot_dog/',hotdog), paste0(root_folder,'/validation/hot_dog'))
#}

#for (not_hotdog in move_not_hotdog){
#    file.move(paste0(root_folder,'/train/not_hot_dog/',not_hotdog), paste0(root_folder,'/validation/not_hot_dog'))
#}

# Check one image 

example_image_path <- file.path("C:/Workspace/R_directory/CEU/machine-learning-course/data/hot-dog-not-hot-dog/train/hot_dog/2417.jpg")
image_read(example_image_path)  # this is a PIL image
img <- image_load(example_image_path, target_size = c(150, 150))

x <- image_to_array(img) / 255
grid::grid.raster(x)

# Preprocess data


train_datagen <- image_data_generator(rescale = 1/255)  
valid_datagen <- image_data_generator(rescale = 1/255)  
test_datagen <- image_data_generator(rescale = 1/255) 


image_size <- c(150, 150)
batch_size <- 50


train_datagen = image_data_generator(
    rescale = 1/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = TRUE,
    fill_mode = "nearest"
)

train_generator <- flow_images_from_directory(
    file.path(root_folder,'train'), # Target directory  
    train_datagen,              # Data generator
    target_size = image_size,  # Resizes all images to 150 × 150
    batch_size = batch_size,
    class_mode = "binary"       # binary_crossentropy loss for binary labels
)

valid_generator <- flow_images_from_directory(
    file.path(root_folder,'/validation/'), # Target directory  
    valid_datagen,              # Data generator
    target_size = image_size,  # Resizes all images to 150 × 150
    batch_size = batch_size,
    class_mode = "binary"       # binary_crossentropy loss for binary labels
)


test_generator <- flow_images_from_directory(
    file.path(root_folder,'/test/'),
    test_datagen,
    target_size = image_size,
    batch_size = batch_size,
    class_mode = "binary"
)


hotdog_model <- keras_model_sequential() 

hotdog_model %>% 
    layer_conv_2d(filters = 32,
                  kernel_size = c(3, 3), 
                  activation = 'relu',
                  input_shape = c(150, 150, 3)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
    layer_conv_2d(filters = 16,
                  kernel_size = c(3, 3), 
                  activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
    layer_conv_2d(filters = 16,
                  kernel_size = c(3, 3), 
                  activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
    layer_dropout(rate = 0.25) %>% 
    layer_flatten() %>% 
    layer_dense(units = 8, activation = 'relu') %>% 
    layer_dense(units = 1, activation = "sigmoid")


hotdog_model %>% compile(
    loss = "binary_crossentropy",
    optimizer = 'rmsprop',
    metrics = c("accuracy")
)

histroy <- hotdog_model %>% fit(
    train_generator,
    steps_per_epoch = 398/batch_size,
    epochs = 5,
    validation_data = valid_generator,
    validation_steps = 150/batch_size
)

# Augmentation techniques: 

train_datagen = image_data_generator(
    rescale = 1/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2, # crops corners of most photos --> useful!
    horizontal_flip = TRUE,
    fill_mode = "nearest"
)

train_generator <- flow_images_from_directory(
    file.path(root_folder,'train'), # Target directory  
    train_datagen,              # Data generator
    target_size = image_size,  # Resizes all images to 150 × 150
    batch_size = batch_size,
    class_mode = "binary"       # binary_crossentropy loss for binary labels
)


history<- hotdog_model %>% fit(
    train_generator,
    steps_per_epoch = 398/batch_size,
    epochs = 5,
    validation_data = valid_generator,
    validation_steps = 1
)


plot(history)
# let's use pretrained models - Inception V3 is suggested, but let's use mobilenet first
# https://blogs.rstudio.com/ai/posts/2017-12-14-image-classification-on-small-datasets/
# https://github.com/stratospark/food-101-keras

model_imagenet <- application_mobilenet(weights = "imagenet")

example_image_path <- file.path('C:/Workspace/R_directory/CEU/machine-learning-course/data/hot-dog-not-hot-dog/train/hot_dog/2417.jpg')
img <- image_load(example_image_path, target_size = c(224, 224))  # 224: to conform with pre-trained network's inputs
x <- image_to_array(img)

# ensure we have a 4d tensor with single element in the batch dimension,
# the preprocess the input for prediction using mobilenet
x <- array_reshape(x, c(1, dim(x)))
x <- mobilenet_preprocess_input(x)

# make predictions then decode and print them
preds <- model_imagenet %>% predict(x)
mobilenet_decode_predictions(preds, top = 3)[[1]]



train_datagen = image_data_generator(
    rescale = 1/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = TRUE,
    fill_mode = "nearest"
)



image_size <- c(128, 128)
batch_size <- 100  # for speed up

train_generator <- flow_images_from_directory(
    file.path(root_folder,'train'), # Target directory  
    train_datagen,              # Data generator
    target_size = image_size,  # Resizes all images to 150 × 150
    batch_size = batch_size,
    class_mode = "binary"       # binary_crossentropy loss for binary labels
)

valid_generator <- flow_images_from_directory(
    file.path(root_folder,'/validation/'), # Target directory  
    valid_datagen,              # Data generator
    target_size = image_size,  # Resizes all images to 150 × 150
    batch_size = batch_size,
    class_mode = "binary"       # binary_crossentropy loss for binary labels
)


test_generator <- flow_images_from_directory(
    file.path(root_folder,'/test/'),
    test_datagen,
    target_size = image_size,
    batch_size = batch_size,
    class_mode = "binary"
)


# create the base pre-trained model
imagenet <- application_mobilenet(weights = 'imagenet', include_top = FALSE,
                                    input_shape = c(image_size, 3))
# freeze all convolutional mobilenet layers
freeze_weights(imagenet)

# train only the top layers (which were randomly initialized)

# add our custom layers
predictions <- imagenet$output %>% 
    layer_global_average_pooling_2d() %>% 
    layer_dense(units = 16, activation = 'relu') %>% 
    layer_dense(units = 1, activation = 'sigmoid')

# this is the model we will train
model <- keras_model(inputs = imagenet$input, outputs = predictions)

# compile the model (should be done *after* setting layers to non-trainable)
model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 2e-5),
    metrics = c("accuracy")
)

# train the model
model %>% fit(
    train_generator,
    steps_per_epoch = 398 / batch_size,
    epochs = 3,  # takes long time to train more
    validation_data = valid_generator,
    validation_steps = 50
)


model %>% evaluate(test_generator)
