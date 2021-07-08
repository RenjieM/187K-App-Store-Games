rm(list = ls())
# Identify the best configurations that have a good impact on the popularity of your applications.
# Recommendation
# Firstly please define how you will measure popularity? What is it and how do you represent it in numbers
# Process the data and show with plots the impact of features on popularity.
# Please, use any means for this explanation (correlations, general EDA, model weights, SHAP, Partial plots, etc)
library(dplyr)
library(ranger)
library(h2o)
library(caret)
library(lattice)
library(ggplot2)
library(psych)
library(kernlab)
library(pdp)
library(vip)
library(class)
library(pROC)
library(kernlab)


game <- read.csv('~/Coding/Game/appstore_games_march_2021.csv', stringsAsFactors = F)

head(game, 3)
dim(game)
str(game)
summary(game)

game$X[1:10]
game$X <- NULL
game$id <- NULL
game$url <- NULL
game$icon_url <- NULL
# Delete unhelpful columns

for (i in 1:20){
  print(i)
  print(any(is.na(game[ ,i])))
}
# Only number 12 column has null value

length(which(is.na(game$version_count)))
game$version_count <- NULL
# Too much missing value, omit the feature

for (i in c('developer', 'age_rating', 'price', 'purchases', 'languages', 'category', 'sub_category')){
  print(i)
  print(length(table(game[ ,i])))
}

game$age_rating_4 <- ifelse(game$age_rating == '4+', 1, 0)
game$age_rating_9 <- ifelse(game$age_rating == '9+', 1, 0)
game$age_rating_12 <- ifelse(game$age_rating == '12+', 1, 0)
game$age_rating_17 <- ifelse(game$age_rating == '17+', 1, 0)
game$age_rating <- NULL

sort(table(game$price))
length(which(game$price == ''))

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

game$price[which(game$price == '')] <- getmode(game$price)
# Impute unknown value

game$price_f <- ifelse(game$price == 'Free', 1, 0)
game$price_nf <- ifelse(game$price != 'Free', 1, 0)
game$price <- NULL
# There are too much 'free' games, so turn it to categorical instead of numeric

table(game$purchases)
length(which(game$purchases == ''))
game$purchases[which(game$purchases == '')] <- getmode(game$purchases)

game$purchases_y <- ifelse(game$purchases == 'Yes', 1, 0)
game$purchases_n <- ifelse(game$purchases == 'No', 1, 0)
game$purchases <- NULL

sort(table(game$languages), decreasing = T)[1:5]
length(which(game$languages == ''))
game$languages[which(game$languages == '')] <- getmode(game$languages)

game$languages_eng <- ifelse(game$languages == 'English', 1, 0)
game$languages_multiple <- ifelse(game$languages != 'English', 1, 0)
game$languages <- NULL

table(game$category)
game$category <- NULL
# Only one category, delete it

table(game$sub_category)
game$sub_category <- NULL
# Lots of unknown values plus unimportant feature, delete it

for (i in 1:length(game$rating_count)){
  if(substring(game$rating_count[i], nchar(game$rating_count[i])) == 'K'){
    game$rating_count[i] <- as.numeric(substring(game$rating_count[i], 1, nchar(game$rating_count[i])-1))*1000
  } else if(substring(game$rating_count[i], nchar(game$rating_count[i])) == 'M'){
    game$rating_count[i] <- as.numeric(substring(game$rating_count[i], 1, nchar(game$rating_count[i])-1))*1000000
  }
}
game$rating_count <- as.numeric(game$rating_count)
# Switch rating_count column to numeric

game <- subset(game, game$size != '')

for (i in 1:length(game$size)){
  if(substring(game$size[i], nchar(game$size[i])-1, nchar(game$size[i])) == 'GB'){
    game$size[i] <- as.numeric(substring(game$size[i], 1, nchar(game$size[i])-2))*1024
  } else if(substring(game$size[i], nchar(game$size[i])-1, nchar(game$size[i])) == 'MB'){
    game$size[i] <- as.numeric(substring(game$size[i], 1, nchar(game$size[i])-2))
  } else{
    game$size[i] <- as.numeric(substring(game$size[i], 1, nchar(game$size[i])-2))/1024
  }
}
game$size <- as.numeric(game$size)
# Turn size column to numeric in MB level

game$name_app <- NULL
game$developer <- NULL
game$release_date <- NULL
game$last_version_date <- NULL
# Remove useless features, and data clean work is done

ggplot(game, aes(rating_avg)) + 
  geom_histogram(binwidth = .05)

summary(game$rating_count)
ggplot(game[which(game$rating_count <= 12), ], aes(rating_count)) + 
  geom_boxplot()

summary(game$size)
ggplot(game[which(game$size <= 200), ], aes(size)) + 
  geom_boxplot()

str(game)
library(psych)
pairs.panels(game[,c('rating_avg', 'rating_count', 'size', 'price_f', 'purchases_y', 'languages_eng')])
cor(game[,c('rating_avg', 'rating_count', 'size', 'price_f', 'purchases_y', 'languages_eng')])


game_kmeans <- game

normalize <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}
game_kmeans[ ,c('size', 'rating_avg', 'rating_count')] <- as.data.frame(lapply(game_kmeans[ ,c('size', 'rating_avg', 'rating_count')], normalize))
# Could not use scale here because there are some huge outliers

set.seed(2012)
game_kmeans_model <- kmeans(game_kmeans, 5)
game_kmeans_model$size
game_kmeans_model$centers

game$cluster <- game_kmeans_model$cluster
aggregate(data=game, rating_count~cluster, mean)
aggregate(data=game, rating_avg~cluster, mean)
# As the outcome shows, cluster 5 has largest rating avg and rating counts and cluster 2 follows. Hence, I'll label each cluster base on the values of combination of rating_count and rating_avg. Cluster 5 is 'extremely popular', cluster 2 is 'popular', cluster 1&4 are 'normal' and cluster 3 is 'unpopular'
# On top of that, there are some significant features can be discovered for extremely popular cluster. 1. Free 2. Has purchase items inside the game 3. Supported by multiple languages

game$cluster[which(game$cluster == 5)] <- 'extremely popular'
game$cluster[which(game$cluster == 2)] <- 'popular'
game$cluster[which(game$cluster == 1 | game$cluster == 4)] <- 'normal'
game$cluster[which(game$cluster == 3)] <- 'unpopular'
game$cluster <- as.factor(game$cluster)
# Assign label to each cluster

head(game, 5)
game$rating_count <- NULL
game$rating_avg <- NULL
# Since the label is found by rating, prediction model will not include them

set.seed(2021)
sample_val <- sample.int(n = nrow(game), size = floor(0.25*nrow(game)), replace = F)
game_val <- game[sample_val, ]
game2 <- game[-sample_val, ]
# Keep 0.25 data for final validation


set.seed(2021)
sample_2 <- sample.int(n = nrow(game2), size = floor(0.75*nrow(game2)), replace = F)
game2_train_rf <- game2[sample_2, ]
game2_test_rf <- game2[-sample_2, ]
  
n <- length(setdiff(names(game2), 'cluster'))

game2_rf <- ranger(cluster~., data=game2_train_rf, seed=2012)
game2_rf$prediction.error
game2_rf_pred <- predict(game2_rf, game2_test_rf[,1:17])
1- game2_rf$prediction.error
confusionMatrix(game2_rf_pred$predictions, game2_test_rf$cluster)$overall[1] < 2
game2_test_rf[which(game2_rf_pred$predictions != game2_test_rf$cluster), ]
game2_rf$prediction.error

as.vector(game2_rf_pred$predictions)
# The accuracy is 99 percent, 24 prediction mistakes are made by random forest with default parameters

hyper_grid <- expand.grid(
  mtry = floor(n*c(.05, .15, .25, .35, .40)),
  min.node.size = c(1, 3, 5, 10),
  replace = c(TRUE, FALSE),
  sample.fraction = c(.5, .6, .8)
)

for (i in seq_len(nrow(hyper_grid))){
  fit <- ranger(
    formula = cluster~.,
    data = game2_train_rf,
    num.trees = n*10,
    mtry = hyper_grid$mtry[i],
    min.node.size = hyper_grid$min.node.size[i],
    replace = hyper_grid$replace[i],
    sample.fraction = hyper_grid$sample.fraction[i],
    verbose = FALSE,
    seed = 123)
  hyper_grid$error[i] <- fit$prediction.error
}

hyper_grid[hyper_grid$error == min(hyper_grid$error), ]
# Choose one of rows as best parameter combination, then fit data in 

game2_rf_2 <- ranger(cluster~., 
                   data=game2_train_rf, 
                   num.trees = n*10,
                   mtry = 6,
                   min.node.size = 1,
                   replace = TRUE,
                   sample.fraction = 0.6,
                   verbose = FALSE,
                   seed=123)
game2_rf_2$prediction.error
game2_rf_pred_2 <- predict(game2_rf, game2_test_rf[,1:17])
confusionMatrix(game2_rf_pred_2$predictions, game2_test_rf$cluster)
# The accuracy improved, 20 prediction mistakes are made instead


game2_norm <- game2
game2_norm$size <- normalize(game2$size)

game2_norm_train <- game2_norm[sample_2, ]
game2_norm_test <- game2_norm[-sample_2, ]

cv <- trainControl(method = 'repeatedcv', number = 10, repeats = 5)
# 10*5 crass validation
game2_norm_knn <- train(cluster~., 
                        data=game2_norm_train, 
                        method = 'knn', 
                        trControl = cv, 
                        tuneGrid = expand.grid(k = c(5, 10, 15)))
game2_norm_knn$metric
# The final value used for the model was k = 10.

game2_norm_knn_pred <- predict(game2_norm_knn, game2_norm_test[, 1:17])
confusionMatrix(game2_norm_knn_pred, game2_norm_test[, 18])
# The accuracy of KNN model with k equal to 10 is 0.99, there are 22 wrong predictions, which are more than random forest algorithm


set.seed(2012)
game2_norm_svm <- train(cluster ~., 
                        data=game2_norm_train, 
                        method='svmRadial', 
                        trControl = trainControl(method = 'cv', number = 5), 
                        tuneLength = 5)
game2_norm_svm
ggplot(game2_norm_svm) + theme_light()
# As the summary and plot of the model shows, when c equal to 2 the model has biggest accuracy 

game2_norm_svm_pred <- predict(game2_norm_svm, game2_norm_test[, 1:17])
confusionMatrix(game2_norm_svm_pred, game2_norm_test[, 18])
# Base on the confusion matrix tells, the accuracy is 0.99 and there 28 wrong predictions
# Thus, the ensemble function would be built on SVM and random forest algorithm
game2_norm_svm$
a <- ksvm(cluster ~., data=game2_norm_train, kernal='rbfdot', C=2)
b <- predict(a, game2_norm_test[, 1:17])
b


game2_ensemble <- function(formula, traindata, data){
  n <- length(setdiff(names(traindata), 'cluster'))
  game2_en_rf <- ranger(formula, 
                        data=traindata, 
                        num.trees = n*10,
                        mtry = 6,
                        min.node.size = 1,
                        replace = TRUE,
                        sample.fraction = 0.6,
                        verbose = FALSE,
                        seed=123)
  game2_en_rf_acc <- 1 - game2_en_rf$prediction.error
  # Get accuracy from random forest algorithm
  r <- 1:round(.7*nrow(traindata))
  game2_train_trans <- traindata[r, ]
  game2_train_trans$size <- normalize(game2_train_trans$size)
  game2_test_trans <- traindata[-r, ]
  game2_test_trans$size <- normalize(game2_test_trans$size)
  # Transform numeric features 
  set.seed(2012)
  game2_en_svm <- ksvm(cluster ~., 
                       data=game2_train_trans, 
                       kernal='rbfdot', 
                       C=2)
  game2_en_svm_pred <- predict(game2_en_svm, game2_test_trans[, 1:17])
  game2_en_svm_cm <- confusionMatrix(game2_en_svm_pred, game2_test_trans$cluster)
  game2_en_svm_acc <- game2_en_svm_cm$overall[1]
  # Get accuracy from svm algorithm
  if(abs(game2_en_rf_acc - game2_en_svm_acc) <= 0.05){
    r2 <- 1:round(.5*nrow(data))
    data1_1 <- data[r2, ]
    data1_2 <- data[-r2, ]
    data1_2$size <- normalize(data1_2$size)
    prediction1_1 <- predict(game2_en_rf, data1_1)
    prediction1_2 <- predict(game2_en_svm, data1_2)
    final_prediction1 <- c(as.vector(prediction1_1$predictions), as.vector(prediction1_2))
    return(final_prediction1)
  }else if(0.05 < abs(game2_en_rf_acc - game2_en_svm_acc) & abs(game2_en_rf_acc - game2_en_svm_acc) <= 0.10) {
    if(game2_en_rf_acc > game2_en_svm_acc){
      r3 <- 1:round(.7*nrow(data))
      data2_1 <- data[r3, ]
      data2_2 <- data[-r3, ]
      data2_2$size <- normalize(data2_2$size)
      prediction2_1 <- predict(game2_en_rf, data2_1)
      prediction2_2 <- predict(game2_en_svm, data2_2)
      final_prediction2_1 <- c(as.vector(prediction2_1$predictions), as.vector(prediction2_2))
      return(final_prediction_2_1)
    }else{
      r3 <- 1:round(.7*nrow(data))
      data2_1 <- data[r3, ]
      data2_2 <- data[-r3, ]
      data2_1$size <- normalize(data2_1$size)
      prediction2_1 <- predict(game2_en_svm, data2_1)
      prediction2_2 <- predict(game2_en_rf, data2_2)
      final_prediction2_2 <- c(as.vector(prediction2_1), as.vector(prediction2_2$predictions))
      return(final_prediction2_2)
    }
  }else{
    if(game2_en_rf_acc > game2_en_svm_acc){
      prediction3_1 <- predict(game2_en_rf, data)
      return(prediction3_1$predictions)
    }else{
      data3 <- data
      data3$size <- normalize(data3$size)
      prediction3_2 <- predict(game2_en_svm, data3)
      return(prediction3_2)
    }
  }
}



game2_en_pred <- game2_ensemble(cluster~., traindata=game2_train_rf, data=game2_test_rf)
confusionMatrix(as.factor(game2_en_pred), game2_test_rf$cluster)
# Accuracy is 0.99, 34 mistakes have been made

game_final_val <- predict(game2_rf_2, game_val[, 1:17])
confusionMatrix(game_final_val$predictions, game_val$cluster)














