
#Clean environment 

rm(list = ls())


# import packages
library(tidyverse)
library(datasets)
library(MASS)
library(ISLR)
library(caret)
library(viridis)
library(GGally)
library(moments)

df <- readRDS(url('http://www.jaredlander.com/data/manhattan_Train.rds')) %>%
    mutate(logTotalValue = log(TotalValue)) %>%
    drop_na()

# goal will be to predict the logTotalValue - We look at property values in Manhatten
pf <- df %>% as.data.frame()


typeof(pf)
colSums(is.na(df))
# there are no NAs

hist(df$TotalValue)
hist(df$logTotalValue)
# Total value is skewed with a long right tail - log Total value looks closer to normal distr
# I didn't see any extreme values - I won't do any further cleaning 


# EDA ---------------------------------------------------------------------

# Starting with numerical cols 

num_cols <- colnames(df)[sapply(df,is.numeric)] 
qual_cols <- colnames(df)[!colnames(df) %in% num_cols] 
remove <- c('ID', 'TotalValue')
num_cols <- num_cols[!num_cols %in% remove]

str(df[num_cols])

# Field descriptions avaialble here: https://github.com/NYCPlanning/db-pluto/blob/master/metadata/Data_Dictionary.md
# Data is on tax lot level
# I noticed that the following numerical vars are in fact location based vars and should be factors instead of num
#   Council should be factor
#   PolicePrct should be factor
#   Health Area should be converted to factor based on 
df <- 
df %>% mutate(
    Council = factor(Council),
    PolicePrct = factor(PolicePrct),
    HealthArea = factor(HealthArea)
    
)

qual_cols <- c(qual_cols, 'Council', 'PolicePrct','HealthArea')
num_cols <- num_cols[!num_cols %in% c('Council', 'PolicePrct','HealthArea')]
# Based on high level check the following grouppings were made 

areas = c('LotArea', 'BldgArea', 'ComArea', 'ResArea', 'OfficeArea', 'RetailArea', 'GarageArea', 'StrgeArea', 'FactryArea', 'OtherArea')
characteristics = c('NumBldgs','NumFloors', 'UnitsRes', 'UnitsTotal', 'LotFront', 'LotDepth', 'BldgFront', 'BldgDepth')
ratios = c('BuiltFAR','ResidFAR','CommFAR','FacilFAR')
outcome = 'logTotalValue'

# Secondly, the qualitative cols 

sapply(df[qual_cols],unique)

ggplot(df, aes(x = SchoolDistrict, fill = FireService)) +
    geom_bar() + 
    scale_fill_manual(name='Legend - Fire Service',
                       values=c('navyblue','purple4', 'gold')) + 
    theme_bw()


ggplot(df, aes(x = OwnerType, fill = IrregularLot)) +
    geom_bar() + 
    scale_fill_manual(name='Legend - Irregular Lot',
                      values=c('navyblue','purple4', 'gold', 'darkgreen')) + 
    theme_bw()

ggplot(df, aes(x = LotType, fill = Landmark)) +
    geom_bar() + 
    scale_fill_manual(name='Legend - Landmark',
                      values=c('navyblue','purple4', 'gold', 'darkgreen')) + 
    theme_bw()

ggplot(df, aes(x = Built, fill = HistoricDistrict)) +
    geom_bar() + 
    scale_fill_manual(name='Legend - Historic district',
                      values=c('navyblue','purple4', 'gold', 'darkgreen')) + 
    theme_bw()

ggplot(df, aes(x = Proximity, fill = factor(High))) +
    geom_bar() + 
    scale_fill_manual(name='Legend - High',
                      values=c('navyblue','purple4', 'gold', 'darkgreen')) + 
    theme_bw()

ggplot(df, aes(x = Built, fill = factor(High))) +
  geom_bar() + 
  scale_fill_manual(name='Legend - High',
                    values=c('navyblue','purple4', 'gold', 'darkgreen')) + 
  theme_bw()

# I will only keep ZoneDist1 and 2 since there are too many missing values for the rest 
for (i in c('ZoneDist1' , 'ZoneDist2', 'ZoneDist3' , 'ZoneDist4')){
    print(paste0('Zone',i,'has this many missing rows: ',df[i] %>% filter(.=='Missing') %>% nrow))
}


location = c('SchoolDistrict', 'ZoneDist1' , 'ZoneDist2', 'Proximity', 'HistoricDistrict', 'Council', 'PolicePrct', 'HealthArea')


characteristics = c(characteristics, 'Class', "LandUse", 'OwnerType', 'Extension', 'IrregularLot',
                    'LotType', 'BasementType', 'Landmark', 'Built', 'High')

# I will also try out a few interactions drawing conclusoins from the plots and a few handpicked

interactions = c('Built * HistoricDistrict', 'Proximity * High', 'Built * High')

# Feature engineering -----------------------------------------------------

# look at skew of vars 
source('https://raw.githubusercontent.com/zsomborh/airbnb_lisbon/main/code/helper.R')

skews <- sapply(num_cols,function(x) {
  skewness(df[,x],na.rm = T)
})

skews

# The following variables are very skewed - they have 1 extreme variable 
# Which I will remove as it is suspicious
df %>%  arrange(desc(RetailArea)) %>% slice(1:10)  %>%dplyr::select(RetailArea)
df %>%  arrange(desc(BldgDepth)) %>% slice(1:10)  %>%dplyr::select(BldgDepth)
df %>%  arrange(desc(GarageArea)) %>% slice(1:10)  %>%dplyr::select(GarageArea)
df %>%  arrange(desc(OfficeArea)) %>% slice(1:10)  %>%dplyr::select(OfficeArea)
df<- df %>% filter(!RetailArea == max(RetailArea)) %>% filter(!BldgDepth == max(BldgDepth)) %>% filter(!GarageArea == max(GarageArea))

skews <- sapply(num_cols,function(x) {
  skewness(df[,x],na.rm = T)
})

skews

# take logs of skewed variables (abs skewness > 3)

to_ln <- sub("\\..*", "", names(skews[abs(skews) > 3]))
df <- get_lns(df,num_cols,3)
ln_vars <- paste0('ln_', to_ln)

df %>% data.frame %>% typeof

for (i in ln_vars){
  df[!is.finite(unlist(df[i])),i] <- 0
}

# Modeling  ---------------------------------------------------------------

# Separete train and holdout set, keep 30% as training 

set.seed(8)
train_indices <- as.integer(createDataPartition(df$logTotalValue, p = 0.7, list = FALSE))
df_train <- df[train_indices, ]
df_holdout <- df[-train_indices, ]

#set train control : 10-fold cv

train_control <- trainControl(
    method = "cv",
    number = 10,
    savePredictions = TRUE,
    verboseIter = FALSE
)

# run OLS with increasing complexity 

ols_model1 <- train(
    formula(paste0("logTotalValue ~", paste0(areas, collapse = " + "))),
    data = df_train,
    method = "lm",
    preProcess = c('center', 'scale'),
    trControl = train_control
)


ols_model2 <- train(
  formula(paste0("logTotalValue ~", paste0(c(areas,characteristics), collapse = " + "))),
  data = df_train,
  method = "lm",
  preProcess = c('center', 'scale'),
  trControl = train_control
)


ols_model3 <- train(
  formula(paste0("logTotalValue ~", paste0(c(areas,characteristics,ratios), collapse = " + "))),
  data = df_train,
  method = "lm",
  preProcess = c('center', 'scale'),
  trControl = train_control
)


ols_model4 <- train(
  formula(paste0("logTotalValue ~", paste0(c(areas,characteristics,ratios, location), collapse = " + "))),
  data = df_train,
  method = "lm",
  preProcess = c('center', 'scale'),
  trControl = train_control
)


final_predictors = c( areas,characteristics,ratios, location,interactions, ln_vars)
final_predictors = final_predictors[!final_predictors %in% to_ln]


ols_model5 <- train(
  formula(paste0("logTotalValue ~", paste0(final_predictors, collapse = " + "))),
  data = df_train,
  method = "lm",
  preProcess = c('center', 'scale'),
  trControl = train_control
)

ols_model5$results[c('RMSE', 'Rsquared')]

ols_models <- list(
  'OLS1' = ols_model1,
  'OLS2' = ols_model2,
  'OLS3' = ols_model3,
  'OLS4' = ols_model4,
  'OLS5' = ols_model5
)

ols_cv_results <- resamples(ols_models) %>%  summary()
ols_cv_results$statistics$RMSE[,'Mean']
ols_cv_results$statistics$Rsquared[,'Mean']

#checking on holdout
result_holdout <- map(ols_models, ~{
  RMSE(predict(.x, newdata = df_holdout), df_holdout[["logTotalValue"]])
}) %>% unlist() %>% as.data.frame() 

result_holdout


# Run penalised models ----------------------------------------------------
# 1st one is a Lasso 


lasso_tune_grid <- expand.grid(
  "alpha" = c(1),
  "lambda" = seq(0.005, 0.2, by = 0.01)
)

set.seed(7)
lasso_model <- train(
  formula(paste0("logTotalValue ~", paste0(final_predictors, collapse = " + "))),
  data = df_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = lasso_tune_grid,
  trControl = train_control
)


# 2nd is the ridge model

ridge_tune_grid <- expand.grid(
  "alpha" = c(0),
  "lambda" = seq(0.005, 0.2, by = 0.01)
)

set.seed(7)
ridge_model <- train(
  formula(paste0("logTotalValue ~", paste0(final_predictors, collapse = " + "))),
  data = df_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = ridge_tune_grid,
  trControl = train_control
)


# 3rd is elastic net - I will try small lambdas as it was appareantly not 

enet_tune_grid <- expand.grid(
  "alpha" = seq(0, 1, by = 0.1),
  "lambda" = seq(0.005, 0.2, by = 0.01)
)

set.seed(7)
enet_model <- train(
  formula(paste0("logTotalValue ~", paste0(final_predictors, collapse = " + "))),
  data = df_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = enet_tune_grid,
  trControl = train_control
)

ggplot(enet_model)

saveRDS(enet_model, 'enet.rds')

# Comparing models 

lasso_model$bestTune
ridge_model$bestTune
enet_model$bestTune


penalised_models <- list(
  'OLS' = ols_model5,
  'Lasso' = lasso_model,
  'Ridge' = ridge_model,
  'Elastic Net' = enet_model
)

penalised_cv_results <- resamples(penalised_models) %>%  summary()
penalised_cv_results$statistics$RMSE[,'Mean']


#Model evaluation 
penalised_result_holdout
bwplot(resamples(penalised_models))

model_differences <- diff(resamples(penalised_models))
dotplot(model_differences)

#checking on holdout - for the sake of curiosity
penalised_result_holdout <- map(penalised_models, ~{
  RMSE(predict(.x, newdata = df_holdout), df_holdout[["logTotalValue"]])
}) %>% unlist() %>% as.data.frame() 

# simplest model that is still good enough ()

train_control_onese <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = TRUE,
  verboseIter = FALSE,
  selectionFunction = "oneSE"
)

set.seed(7)
enet_model_onese <- train(
  formula(paste0("logTotalValue ~", paste0(final_predictors, collapse = " + "))),
  data = df_train,
  method = "glmnet",
  preProcess = c('center', 'scale'),
  tuneGrid = enet_tune_grid,
  trControl = train_control_onese
)

resamples(list(enet_model, enet_model_onese)) %>% summary()

enet_model$bestTune
enet_model_onese$bestTune


# Using PCA on linear models ----------------------------------------------

# We first try to see the optimal number of components with pcr 
# Due to size limitations this is run on the train set only 

vars_used <- length(ols_model5$coefnames)

tune_grid <- data.frame(ncomp = 60:vars_used)
set.seed(7)
pcr_fit <- train(
  formula(paste0("logTotalValue ~", paste0(final_predictors, collapse = " + "))) ,
  data = df_train,
  method = "pcr",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = tune_grid,
  preProcess = c("center", "scale")
)

ggplot(pcr_fit) +xlim(60,236) + ylim(0.52,0.65) 

mean(ols_model5$resample$RMSE)
mean(pcr_fit$resample$RMSE)
mean(pcr_fit$resample$Rsquared)

# Using PCA for preprocessing for elastic net: 

pcr_fit$bestTune[[1]]


train_control_enet_pca <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = TRUE,
  verboseIter = FALSE,
  preProcOptions = list(pcaComp =pcr_fit$bestTune[[1]])
)

enet_pca_tune_grid <- expand.grid(
  "alpha" = seq(0, 0.2, by = 0.05),
  "lambda" = seq(0.005, 0.05, by = 0.01)
)

enet_model$bestTune

set.seed(7)
enet_model_pca <- train(
  formula(paste0("logTotalValue ~", paste0(final_predictors, collapse = " + "))),
  data = df_train,
  method = "glmnet",
  preProcess = c("center", "scale", 'pca', 'nzv'),
  tuneGrid = enet_pca_tune_grid,
  trControl = train_control_enet_pca
)

enet_model_pca


# Evaluate best model on holdout ------------------------------------------

RMSE(predict(enet_model_pca, newdata = df_holdout), df_holdout[["logTotalValue"]])
