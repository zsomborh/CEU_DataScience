
# Preparation -------------------------------------------------------------



library(tidyverse)
library(data.table)
library(h2o)
h2o.init()

train_df <- fread('Kaggle/train.csv')
test_df <- fread('Kaggle/test.csv')

source('C:/Users/T450s/Desktop/programming/git/CEU_DataScience/Data_Science_2-homework_1/helper.R')


my_seed <- 20210409

train_df$is_popular<- factor(ifelse(train_df$is_popular == 1, 'yes', 'no'), levels = c('yes', 'no'))


data_split <- h2o.splitFrame(as.h2o(train_df), ratios = c(0.8), seed = my_seed)
data_train <- data_split[[1]]
data_valid<- data_split[[2]]



y <- 'is_popular'
X <- setdiff(colnames(data_train),c(y))


# Modeling ----------------------------------------------------------------



rf_model <- h2o.randomForest(
    X, y,
    training_frame = data_train,
    model_id = "rf",
    ntrees = 500,
    mtries =1,
    max_depth = 1,
    seed = my_seed,
    nfolds = 5,
    keep_cross_validation_predictions = TRUE
)


gbm_model <- h2o.gbm(
    X, y,
    training_frame = data_train,
    model_id = "gbm",
    ntrees = 450,
    max_depth = 4,
    learn_rate = 0.012,
    seed = my_seed,
    nfolds = 5,
    keep_cross_validation_predictions = TRUE
)

deeplearning_model <- h2o.deeplearning(
    X, y,
    training_frame = data_train_all,
    model_id = "deeplearning",
    hidden = c(128,50,30),
    seed = my_seed,
    epochs = 200,
    epsilon  = 1e-10,
    max_w2 = 100,
    nfolds = 5,
    input_dropout_ratio = 0.1,
    keep_cross_validation_predictions = TRUE,
    stopping_metric = 'AUC'
)

my_models <- list(rf_model, gbm_model, deeplearning_model)
all_performance <- map_df( my_models, getPerformanceMetrics, xval = TRUE)
plotROC(all_performance)

all_performance

validation_performances <- list(
    #"simple_lm" = h2o.auc(h2o.performance(simple_lm, newdata = data_valid)),
    "rf" = h2o.auc(h2o.performance(rf_model, newdata = data_valid)),
    "gbm" = h2o.auc(h2o.performance(gbm_model, newdata = data_valid)),
    "deeplearning" = h2o.auc(h2o.performance(deeplearning_model, newdata = data_valid))
)
validation_performances


# Deeplearning submission
h2o.auc(h2o.performance(deeplearning_model))

pred <- h2o.predict(deeplearning_model, data_test)
to_submit<- cbind(data_test %>% as.data.frame() %>% select(article_id), pred$yes %>% as.data.frame())
colnames(to_submit) <- c('article_id', 'score')

write_csv(to_submit, 'Kaggle/submission20.csv')


# AUTOML!

#let's train it on the whole dataset
data_train_all <- as.h2o(train_df)

aml <- h2o.automl(x = X, y = y,
                  training_frame = data_train_all,
                  max_models = 15,
                  seed = my_seed)

aml@leaderboard

h2o.auc(h2o.performance(aml@leader, newdata = data_valid))

aml@leaderboard %>% as.data.frame()

# saving predictions + models

data_test <- as.h2o(test_df)

pred <- h2o.predict(aml@leader, data_test)
to_submit<- cbind(data_test %>% as.data.frame() %>% select(article_id), pred$yes %>% as.data.frame())
colnames(to_submit) <- c('article_id', 'score')

write_csv(to_submit, 'Kaggle/submission17.csv')

model_path <- h2o.saveModel(object = aml@leader, path = getwd(), force = TRUE)

saveRDS(as.data.frame(aml@leaderboard), 'aml_leaderboard.rds')
