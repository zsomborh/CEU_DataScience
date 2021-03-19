rm(list = ls())

library(tidyverse)
theme_set(theme_bw())

library(h2o)
h2o.init()


source('C:/Users/T450s/Desktop/programming/git/CEU_DataScience/Data_Science_2-homework_1/helper.R')

## Get the data 

data <- read_csv("Data_Science_2-homework_1/KaggleV2-May-2016.csv")

# some data cleaning
data <- select(data, -one_of(c("PatientId", "AppointmentID", "Neighbourhood"))) %>%
    janitor::clean_names()

# for binary prediction, the target variable must be a factor + generate new variables
data <- mutate(
    data,
    no_show = factor(no_show, levels = c("Yes", "No")),
    handcap = ifelse(handcap > 0, 1, 0),
    across(c(gender, scholarship, hipertension, alcoholism, handcap), factor),
    hours_since_scheduled = as.numeric(appointment_day - scheduled_day),
    diabetes = factor(diabetes),
    age_sq = age **2,
    hours_since_scheduled_ln = log(hours_since_scheduled+1)
)

# clean up a little bit
data <- filter(data, between(age, 0, 95), hours_since_scheduled >= 0) %>%
    select(-one_of(c("scheduled_day", "appointment_day", "sms_received")))


h2o.no_progress()

my_seed <- 20210318

skimr::skim(data)


data_split <- h2o.splitFrame(as.h2o(data), ratios = c(0.05,0.45), seed = my_seed)
data_train <- data_split[[1]]
data_valid<- data_split[[2]]
data_test <- data_split[[3]]

y <- "no_show"
X <- setdiff(names(data_train), c(y, 'hours_since_scheduled'))


nrow(as_tibble(data_test))/nrow(data)

## Train benchmark model 

simple_lm <- h2o.glm(
    X, y,
    training_frame = data_train,
    model_id = "logit",
    lambda = 0,
    nfolds = 5,
    seed = my_seed,
    keep_cross_validation_predictions = TRUE
)

h2o.auc(h2o.performance(simple_lm, newdata =data_valid))
plot(h2o.performance(simple_lm, newdata =data_valid), type = "roc")

## Train other three modesl 


rf_model <- h2o.randomForest(
    X, y,
    training_frame = data_train,
    model_id = "rf",
    ntrees = 500,
    mtries = 3,
    max_depth = 3,
    seed = my_seed,
    nfolds = 5,
    keep_cross_validation_predictions = TRUE
)

gbm_model <- h2o.gbm(
    X, y,
    training_frame = data_train,
    model_id = "gbm",
    ntrees = 700,
    max_depth = 2,
    learn_rate = 0.01,
    seed = my_seed,
    nfolds = 5,
    keep_cross_validation_predictions = TRUE
)

deeplearning_model <- h2o.deeplearning(
    X, y,
    training_frame = data_train,
    model_id = "deeplearning",
    hidden = c(10,10,10,10,10),
    seed = my_seed,
    epochs = 200,
    epsilon  = 1e-10,
    max_w2 = 100,
    nfolds = 5,
    input_dropout_ratio = 0.1,
    keep_cross_validation_predictions = TRUE,
    stopping_metric = 'AUC'
)

## Validate all models

my_models <- list(simple_lm, rf_model, gbm_model, deeplearning_model)
all_performance <- map_df( my_models, getPerformanceMetrics, xval = TRUE)
plotROC(all_performance)


validation_performances <- list(
    "simple_lm" = h2o.auc(h2o.performance(simple_lm, newdata = data_valid)),
    "rf" = h2o.auc(h2o.performance(rf_model, newdata = data_valid)),
    "gbm" = h2o.auc(h2o.performance(gbm_model, newdata = data_valid)),
    "deeplearning" = h2o.auc(h2o.performance(deeplearning_model, newdata = data_valid))
)
validation_performances

## correlations 

h2o.model_correlation_heatmap(my_models, data_valid)
h2o.varimp_heatmap(my_models)

# stacking base learners 

ensemble_model <- h2o.stackedEnsemble(
    X, y,
    training_frame = data_train,
    base_models = my_models,
    keep_levelone_frame = TRUE
)



test_performances <- list(
    "simple_lm" = h2o.auc(h2o.performance(simple_lm, newdata = data_test)),
    "rf" = h2o.auc(h2o.performance(rf_model, newdata = data_test)),
    "gbm" = h2o.auc(h2o.performance(gbm_model, newdata = data_test)),
    "deeplearning" = h2o.auc(h2o.performance(deeplearning_model, newdata = data_test)),
    'ensemble_model' =  h2o.auc(h2o.performance(ensemble_model, newdata = data_test))
)
test_performances


## this is not really required

ensemble_model@model$metalearner_model@model$coefficients_table

map_df(
    c(my_models, ensemble_model),
    ~{tibble(model = .@model_id, auc = h2o.auc(h2o.performance(., newdata = data_test)))}
)

ensemble_model_gbm <- h2o.stackedEnsemble(
    X, y,
    training_frame = data_train,
    metalearner_algorithm = "gbm",
    base_models = my_models
)

h2o.auc(h2o.performance(ensemble_model_gbm, newdata = data_test))
