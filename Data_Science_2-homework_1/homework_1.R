library(tidyverse)
library(h2o)
library(caret)
library(xgboost)
library(rattle)
library(pROC)

rm(list = ls())

source('C:/Users/T450s/Desktop/programming/git/CEU_DataScience/Data_Science_2-homework_1/helper.R')
source('https://raw.githubusercontent.com/zsomborh/business_growth_prediction/main/codes/helper.R')


data <- as_tibble(ISLR::OJ)

h2o.init()
skimr::skim(data)

data <- data %>% mutate(
    WeekofPurchase = factor(WeekofPurchase),
    StoreID = factor(StoreID),
    SpecialCH = factor(SpecialCH),
    SpecialMM = factor(SpecialMM),
    STORE = factor(STORE)
)

h2o_data <- as.h2o(data)

splitted_data <- h2o.splitFrame(h2o_data, ratios = c(0.75), seed = 20210316)
train_data <- splitted_data[[1]]
test_data <- splitted_data[[2]]


response = "Purchase"
predictors = setdiff(
    colnames(data), 
    c(response))


## Decision tree 
dectree = 
    h2o.gbm(x = predictors, y = response, 
            training_frame = train_data, 
            ntrees = 1, min_rows = 1, 
            sample_rate = 1, col_sample_rate = 1,
            max_depth = 4,
            nfolds=5,
            stopping_rounds = 3, stopping_tolerance = 0.01, 
            stopping_metric = "AUC", 
            seed = 20210316)


## this part is for H20 plotting 
titanicH2oTree = h2o.getModelTree(model = dectree, tree_number = 1)

titanicDataTree = createDataTree(titanicH2oTree)

GetEdgeLabel <- function(node) {return (node$edgeLabel)}
GetNodeShape <- function(node) {switch(node$type, 
                                       split = "diamond", leaf = "oval")}
GetFontName <- function(node) {switch(node$type, 
                                      split = 'Palatino-bold', 
                                      leaf = 'Palatino')}
SetEdgeStyle(titanicDataTree, fontname = 'Palatino-italic', 
             label = GetEdgeLabel, labelfloat = TRUE,
             fontsize = "26", fontcolor='royalblue4')
SetNodeStyle(titanicDataTree, fontname = GetFontName, shape = GetNodeShape, 
             fontsize = "26", fontcolor='royalblue4',
             height="0.75", width="1")

SetGraphStyle(titanicDataTree, rankdir = "LR", dpi=70.)

plot(titanicDataTree, output = "graph")



## Part 2 

## Rf 

# here ntrees is also a tuning parameter
rf_params <- list(ntrees = c(300,  500), mtries = c(2, 3, 5))

rf_grid <- h2o.grid(
    "randomForest", x = predictors, y = response,
    training_frame = train_data,
    nfolds = 5,
    seed = 20210316,
    hyper_params = rf_params
)

h2o.getGrid(rf_grid@grid_id, "auc")@model_ids
rf_model <- h2o.getModel(h2o.getGrid(rf_grid@grid_id, "auc")@model_ids[[6]])

h2o.rmse(h2o.performance(rf_model, xval=T))
h2o.auc(h2o.performance(rf_model, xval=T))
max(h2o.accuracy(h2o.performance(dectree, xval = T)))

## GBM

gbm_params <- list(
    learn_rate = c(0.01, 0.05, 0.1),  # default: 0.1
    ntrees = c( 100, 300),
    max_depth = c(2, 5),
    sample_rate = c(0.2, 0.8)
)

gbm_grid <- h2o.grid(
    "gbm", x = predictors, y = response,
    grid_id = "gbm",
    training_frame = train_data,
    nfolds = 5,
    seed = 20210316,
    hyper_params = gbm_params
)

best_gbm <- h2o.getModel(
    h2o.getGrid(gbm_grid@grid_id, "auc")@model_ids[[24]]
)

h2o.rmse(h2o.performance(best_gbm, newdata = train_data)) 
h2o.auc(h2o.performance(best_gbm, newdata = train_data))

## XGBoost 
# I will need to use caret here 

train_control <- trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE, 
    summaryFunction = twoClassSummaryExtended,
    savePredictions = TRUE,
    verboseIter = TRUE, 
    allowParallel = TRUE
)


xgb_grid <-  expand.grid(nrounds=c(350,500),
                         max_depth = c(2,3, 4),
                         eta = c(0.03,0.05, 0.06),
                         gamma = c(0.01),
                         colsample_bytree = c(0.5),
                         subsample = c(0.75),
                         min_child_weight = c(0))

train_data_tibble <- as_tibble(train_data)


#train_data_tibble  <- train_data_tibble %>% mutate(Purchase = as.factor(ifelse (Purchase == 'CH',1,0) ))

set.seed(7)
xgb_model <- train(
    formula(paste0("Purchase ~", paste0(predictors, collapse = " + "))),
    method = "xgbTree",
    data = train_data_tibble,
    tuneGrid = xgb_grid,
    trControl = train_control
)


auc <- list()
for (fold in c("Fold1", "Fold2", "Fold3", "Fold4", "Fold5")) {
    cv_fold <-
        xgb_model$pred %>%
        filter(Resample == fold)
    
    roc_obj <- roc(cv_fold$obs, cv_fold$CH)
    auc[[fold]] <- as.numeric(roc_obj$auc)
}
auc_df = data.frame("Resample" = names(auc),"AUC" = unlist(auc))

xgb_acc = mean(xgb_model$resample[,c("Resample", "Accuracy")]$Accuracy)
xgb_rmse = mean(xgb_model$resample[,c("Resample", "RMSE")]$RMSE)
xgb_auc = mean(auc_df$AUC)

# Summarise results

# get dectree results

summary <- list()

models <- c('dectree'  = dectree, 'rf' = rf_model, 'gbm' = best_gbm)

summary_df <- sapply(models,get_scores) %>% as.data.frame()

xgb_list <- list('xgboost' = c(xgb_rmse, xgb_auc,xgb_acc))
summary_df <- cbind(summary_df,xgb_list)

summary_df

# Choosing GBM

plot(h2o.performance(best_gbm, newdata = test_data, xval = T),valid=T,type='roc')
h2o.auc(h2o.performance(best_gbm, newdata = test_data, xval = T))

## Var importance plots 

h2o.varimp_plot(rf_model)
h2o.varimp_plot(best_gbm)
varimp_df <- varImp(xgb_model)
varimp_df <- varimp_df$importance %>% as.data.frame() %>% mutate(
    vars = rownames(varimp_df$importance),
    cat = ifelse(startsWith(vars,'WeekofPurchase'),'WeekofPurchase',
                 ifelse(startsWith(vars,'StoreID'),'StoreID',
                        ifelse(startsWith(vars,'STORE'),'STORE',vars)
                 ))) 


varimp_df <- varimp_df %>% group_by(cat) %>% summarise(imp = sum(Overall)) %>% arrange(desc(imp))
ggplot(varimp_df, aes(x = reorder(cat,imp), y = imp))+geom_bar(stat = 'identity', fill = 'navyblue') + coord_flip() + theme_bw()


saveRDS(rf_grid, 'rfgrid.rds')
saveRDS(gbm_grid, 'gbm_grid.rds')
saveRDS(xgb_model, 'xgb_model.rds')
saveRDS(dectree, 'dectree.rds')
