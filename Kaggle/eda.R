library(tidyverse)
library(data.table)
library(viridis)
train_df <- fread('Kaggle/train.csv') %>% as.data.frame()
train_df <- train_df %>% mutate(is_popular = as.factor(is_popular))

# structure of data
str(train_df)

# Check data quality
colSums(is.na(train_df))

# Around 20 % of population is positive 
sum(train_df$is_popular)/nrow(train_df)


# Define variable sets ----------------------------------------------------


Y <- 'is_popular'

tokens <- c('n_tokens_title', 'n_tokens_content', 'n_unique_tokens',
           'average_token_length', 'n_non_stop_words', 'n_non_stop_unique_tokens'
           )
channel <- colnames(train_df)[substr(colnames(train_df),1,12) == 'data_channel']
dow <- c(colnames(train_df)[substr(colnames(train_df),1,7) == 'weekday'], 'is_weekend')
keywords <- c(colnames(train_df)[substr(colnames(train_df),1,2) == 'kw'], 'num_keywords')

references <- c('num_hrefs','num_self_hrefs','num_imgs','num_videos',
'self_reference_min_shares', 'self_reference_max_shares',
'self_reference_avg_sharess')


sentiment <- c(
  'global_subjectivity','global_sentiment_polarity','global_rate_positive_words',
  'global_rate_negative_words','rate_positive_words','rate_negative_words',
  'avg_positive_polarity','min_positive_polarity','max_positive_polarity',
  'avg_negative_polarity','min_negative_polarity','max_negative_polarity',
  'title_subjectivity','title_sentiment_polarity','abs_title_subjectivity',
  'abs_title_sentiment_polarity'
  )

LDA <- colnames(train_df)[substr(colnames(train_df),1,3) == 'LDA']


X <- setdiff(colnames(train_df),c(Y))#, 'article_id'))



# Plotting histograms and boxplots ----------------------------------------

train_df[tokens]
plot_hists <- function(df, vars) {
    melted <- reshape2::melt(df[vars])
    
    p<- ggplot(melted, aes(x=value)) + 
        geom_histogram()  + facet_wrap(~variable, scales = 'free')
    return(p)
}

melted <- reshape2::melt(train_df[tokens])

plot_hists(train_df, c(references))
plot_hists(train_df, c(tokens))

train_df$n_unique_tokens %>% sort(decreasing = T) %>% head()

#remove outlier
train_df <- train_df %>% filter(!n_unique_tokens == max(n_unique_tokens))

plot_hists(train_df, tokens)
plot_hists(train_df, keywords)
plot_hists(train_df, sentiment)
plot_hists(train_df, LDA)


plot_boxplots <- function(df, vars) {
    
    melted <- reshape2::melt(df[c(vars,'is_popular')],id = 'is_popular' )
    
    p <- ggplot(melted) + 
        geom_boxplot(aes(x = is_popular, y = value, fill = is_popular), outlier.shape = NA) + 
        facet_wrap(~variable, scales= 'free')

    return(p)
    
}

# Inspect boxplots

plot_boxplots(train_df, c(tokens ))
plot_boxplots(train_df, c(channel ))
plot_boxplots(train_df, c(dow ))
plot_boxplots(train_df, c(keywords ))
plot_boxplots(train_df, c(references )) # interesting findings here 
plot_boxplots(train_df, c(sentiment ))
plot_boxplots(train_df, c(LDA )) # Latent Dirichlet Allocation


# Get ln of references

plot_hists(train_df,references)

train_df <- train_df %>% mutate(
    ln_num_hrefs = log(num_hrefs+1),
    ln_num_self_hrefs = log(num_self_hrefs+1),
    ln_num_imgs = log(num_imgs+1)
    
)

# Write transformed dataframe to disk
write_csv(train_df,'Kaggle/train_df_trans.csv')
