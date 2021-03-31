rm(list = ls())

library(tidyverse)
library(caret)
library(skimr)
library(janitor)
library(factoextra) # provides nice functions for visualizing the output of PCA
library(NbClust) # for choosing the optimal number of clusters
library(GGally)
library(knitr)
library(kableExtra)
library(data.table)


# Get and prepare data ----------------------------------------------------

#look at some descriptives
df<- USArrests
skim(df)
ggpairs(df, columns = c("Murder", "Assault", "UrbanPop", "Rape"))

# preprocess vars for clustering 

df <- scale(df) %>% as.data.frame()


# Get optimal number of clusters ------------------------------------------

#look for elbow
fviz_nbclust(df, kmeans, method = "wss")

#look for majority voitng per NBclust
nb <- NbClust(df, method = "kmeans", min.nc = 2, max.nc = 10, index = "all")
nb # beased on this - number of clusters to choose is 2
as.data.frame(t(nb$Best.nc))
  
# K-means -----------------------------------------------------------------
km <- kmeans(df, centers = 2, nstart = 50)
km

df<- df  %>% mutate(cluster = factor(km$cluster))

# some visualisations:
ggplot(df, aes(x = UrbanPop, y = Murder, color = cluster)) +
  geom_point()

ggplot(df, aes(x = UrbanPop, y = Assault, color = cluster)) +
  geom_point()

ggplot(df, aes(x = UrbanPop, y = Rape, color = cluster)) +
  geom_point()

centers <- as_tibble(km$centers) %>% 
  mutate(cluster = factor(seq(nrow(km$centers))), center = TRUE)

data_w_clusters_centers <- bind_rows(df, centers)

ggplot(data_w_clusters_centers, aes(
  x = UrbanPop, y = Murder,
  color = cluster, size = ifelse(!is.na(center), 2, 1))
) +
  geom_point() +
  scale_size(guide = 'none')


# Run PCA and get first two PCs

pca_result <- prcomp(df[,c('Murder', 'Assault', 'UrbanPop', 'Rape')]) # I didn't use scale = T arg as df is already scaled

first_two_pc <- as.data.frame(pca_result$x[,c('PC1', 'PC2')]) %>% mutate(cluster = factor(km$cluster))

ggplot(first_two_pc) + geom_point(aes(x= PC1, y = PC2, colour = cluster))
