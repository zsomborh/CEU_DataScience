library(tidyverse)
library(data.table)
library(caret)
library(xgboost)
library(pROC)

train_df <- fread('Kaggle/train_df_trans.csv')
test_df <- fread('Kaggle/test.csv')
submit_sample <- fread('Kaggle/sample_submission.csv')

Y <- 'is_popular'

#weekdays <- colnames(train_df)[substr(colnames(train_df),1,7) == 'weekday']
#channel <- colnames(train_df)[substr(colnames(train_df),1,12) == 'data_channel']


X <- setdiff(colnames(train_df),c(Y,'num_hrefs', 'num_self_hrefs', 'num_imgs'))#, 'article_id'))


train_df <-
train_df %>% mutate(
    is_popular = factor(is_popular,levels = list(1,0), labels = c('yes','no'))
) %>% as.data.frame()

formula <- as.formula(paste0(Y, '~',paste0(X,collapse = '+')))

# Starting off with logit enet 

train_control <- trainControl(
    method = "repeatedcv",
    number = 5,
    repeats = 4,
    classProbs = TRUE, 
    summaryFunction = twoClassSummary,
    savePredictions = TRUE,
    verboseIter = TRUE, 
    allowParallel = TRUE
)


enet_grid <- expand.grid(
    "alpha" = seq(0, 1, by = 0.1),
    "lambda" = seq(0.005, 0.1, by = 0.01)
)

set.seed(7)
enet_model <- train(formula,
                   method = "glmnet",
                   preprocess = c('center','scale'),
                   data = train_df,
                   tuneGrid = enet_grid,
                   trControl = train_control)


enet_model$bestTune

test_prediction <- predict.train(enet_model, newdata = data_test)
test_truth <- data_test[["no_show"]]

# Let's see tree based models



# RF

tune_grid <- expand.grid(
    .mtry = c(2,5),
    .splitrule = "gini",
    .min.node.size = c(1,5,10)
)


set.seed(7)
rf_model <- train(
    formula,
    data = train_df,
    method = "ranger",
    trControl = train_control,
    tuneGrid = tune_grid,
    preProcess = c('center' , 'scale'),
    importance = "impurity",
    num.trees = 800
)




xgb_grid <-  expand.grid(nrounds=c(450),
                         max_depth = c(6),
                         eta = c( 0.02),
                         gamma = c(0.012),
                         colsample_bytree = c(  0.35),
                         subsample =
                             c(0.75),
                         min_child_weight = c(0))


set.seed(7)
xgb_model <- train(
    formula,
    method = "xgbTree",
    data = train_df,
    tuneGrid = xgb_grid,
    preProcess = c('center', 'scale'),
    trControl = train_control
)

xgb_model$bestTune
xgb_model$results
xgb_model$resample

auc <- list()
for (fold in c("Fold1", "Fold2", "Fold3", "Fold4", "Fold5")) {
    cv_fold <-
        xgb_model$pred %>%
        filter(Resample == fold)
    
    roc_obj <- roc(cv_fold$obs, cv_fold$yes)
    auc[[fold]] <- as.numeric(roc_obj$auc)
}
auc_df <- data.frame("Resample" = names(auc),"AUC" = unlist(auc))



xgb_acc <- mean(xgb_model$resample[,c("Resample", "ROC")]$ROC)
xgb_rmse <- mean(xgb_model$resample[,c("Resample", "RMSE")]$RMSE)
xgb_auc <- mean(auc_df$AUC)
xgb_auc

predict(xgb_model) %>% as_data_frame() %>% group_by(value) %>% summarise(n())
train_df %>% group_by(is_popular) %>% summarise(n())

# final prediction
colnames(submit_sample)

test_df <- test_df %>% mutate(
    ln_num_hrefs = log(num_hrefs+1),
    ln_num_self_hrefs = log(num_self_hrefs+1),
    ln_num_imgs = log(num_imgs+1)
    
)

test_df[,'score'] <- predict(xgb_model, newdata = test_df, type='prob') %>% as_data_frame() %>% select(yes)

submit_df <- test_df[,c('article_id','score')]

write_csv(submit_df, 'Kaggle/submission24.csv')

varImp(rf_model) %>% tail()
