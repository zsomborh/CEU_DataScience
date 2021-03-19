rm(list = ls())

library(tidyverse)
library(h2o)
library(pROC)

## Load data

data <- as_tibble(ISLR::Hitters) %>%
    drop_na(Salary) %>%
    mutate(log_salary = log(Salary), Salary = NULL)


skimr::skim(data)
str(data)

h2o_data <- as.h2o(data)

## Tune Rfs
response = "log_salary"
predictors = setdiff(
    colnames(data), 
    c(response))


rf_params <- list(ntrees = c(500), mtries = c(2, 10))

## Run Rfs

rf_grid <- h2o.grid(
    "randomForest", x = predictors, y = response,
    training_frame = h2o_data,
    seed = 20210317,
    hyper_params = rf_params
)

rf_model_m_2 <- h2o.getModel(rf_grid@model_ids[[1]])
rf_model_m_10 <- h2o.getModel(rf_grid@model_ids[[2]])


h2o.varimp_plot(rf_model_m_2)
h2o.varimp_plot(rf_model_m_10)

## Do the same with GBM

gbm_params <- list(
    learn_rate = c(0.1), 
    ntrees = c(500),
    max_depth = c(5),
    sample_rate = c(0.1, 1)
)

gbm_grid <- h2o.grid(
    "gbm", x = predictors, y = response,
    training_frame = h2o_data,
    seed = 20210317,
    hyper_params = gbm_params
)

gbm_model_m_01 <- h2o.getModel(gbm_grid@model_ids[[1]])
gbm_model_m_1 <- h2o.getModel(gbm_grid@model_ids[[2]])

h2o.varimp_plot(gbm_model_m_01)
h2o.varimp_plot(gbm_model_m_1)

test_performances <- list(
    "simple_lm" = h2o.auc(h2o.performance(simple_lm, newdata = data_test)),
    "rf" = h2o.auc(h2o.performance(rf_model, newdata = data_test)),
    "gbm" = h2o.auc(h2o.performance(gbm_model, newdata = data_test)),
    "deeplearning" = h2o.auc(h2o.performance(deeplearning_model, newdata = data_test)),
    'ensemble_model' =  h2o.auc(h2o.performance(ensemble_model, newdata = data_test))
)