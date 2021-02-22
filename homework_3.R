library(tidyverse)
library(data.table)
library(caret)
library(factoextra)


data <- fread("data/gene_data_from_ISLR_ch_10/gene_data.csv")
data[, is_diseased := factor(is_diseased)]
dim(data)
tail(names(data))

data_features <- copy(data)
data_features[, is_diseased := NULL]

pca_result <- prcomp(data_features, scale. = TRUE)
fviz_pca_ind(pca_result)

screeplot(pca_result, npcs =50 )

top10_contrib <- pca_result$rotation[,'PC1'] %>% abs() %>% sort(decreasing = T) %>% head(10)
top10_contrib

bottom10_contrib <- pca_result$rotation[,'PC1'] %>% abs() %>% sort(decreasing = T) %>% tail(10)
bottom10_contrib

ggplot(data_features)+geom_point(aes(x = measure_502, y = measure_565))
ggplot(data_features)+geom_point(aes(x = measure_502, y = measure_639))
