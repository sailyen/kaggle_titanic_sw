library(ggplot2)
library(moments)
library(reshape2)
library(GGally)
library(missForest)
library(randomForest)
library(gbm)
library(xgboost)
library(dplyr)
library(caret)
library(ROCR)
library(earth)
train <- read.csv("train.csv",header = T, na.strings = c(" ",""))
test <- read.csv("test.csv",header = T, na.strings = c(" ",""))
head(train)
summary(train)
str(train)

train <- train[, -which(names(train) %in% c("PassengerId","Name","Ticket"))]
test <- test[, -which(names(test) %in% c("PassengerId","Name","Ticket"))]
head(train)

###Missing value
sapply(train, function(x) sum(is.na(x)) )
nrow(train)
ncol(train)
train <- train[, -which(names(train) %in% c("Cabin"))]
test <- test[, -which(names(test) %in% c("Cabin"))]
head(train)

###target Y
target <- summary(train$Survived)
target
gt <- ggplot(data = train, aes(x = train$Survived)) + geom_bar(fill="#FF6666") 
gt

# Y and X
#*******************************************************************************

g1 <- ggplot(train, aes(train$Pclass, train$Survived)) + geom_jitter(color = "sky blue")
g1

g2 <- ggplot(train, aes(train$Sex, train$Survived)) + geom_jitter(color = "sky blue")
g2

g3 <- ggplot(train, aes(train$SibSp, train$Survived)) + geom_jitter(color = "sky blue")
g3

g4 <- ggplot(train, aes(train$Parch, train$Survived)) + geom_jitter(color = "sky blue")
g4

g5 <- ggplot(train, aes(train$Embarked, train$Survived)) + geom_jitter(color = "sky blue")
g5
#
g1 <- ggplot(train, aes(y = train$Survived, x = train$Age)) +
  geom_point(size=2, alpha=0.4) +
  stat_smooth(method="loess", colour="sky blue", size=1.5)
g1

g1 <- ggplot(train, aes(y = train$Survived, x = train$Fare)) +
  geom_point(size=2, alpha=0.4) +
  stat_smooth(method="loess", colour="blue", size=1.5)
g1


#variable X categorical
g1 <- ggplot(data = train, aes(x = train$Sex)) + geom_bar(fill="#FF6666") 
g1
g3 <- ggplot(data = train, aes(x = train$Pclass)) + geom_bar(fill="#FF6666") 
g3

dummies <- dummyVars(Survived ~ ., data = train)
head(predict(dummies, newdata = train))
train_pr <- predict(dummies, newdata = train) %>% as.data.frame()
head(train_pr)

dummies_test <- dummyVars(~.,data = test)
head(predict(dummies_test, newdata = test))
test_pr <- predict(dummies_test, newdata = test)%>% as.data.frame()
head(test_pr)

#variable X numeric
g2 <- ggplot(data = train_pr, aes(x = train$Age)) + geom_histogram(bins = 20)
g2
g2_d <- ggplot(data = train_pr, aes(x = train$Age)) + geom_density(alpha = .2)
g2_d
p <- ggplot(train_pr, aes(sample = Age)) + stat_qq()
p

g4 <- ggplot(data = train_pr, aes(x = train$SibSp)) + geom_bar() 
g4
p <- ggplot(train_pr, aes(sample = SibSp)) + stat_qq()
p
###log transformation
train_pr$SibSp <- log(train_pr$SibSp+1)
test_pr$SibSp <- log(test_pr$SibSp+1)

g5 <- ggplot(data = train_pr, aes(x = train$Parch)) + geom_bar() 
g5
p <- ggplot(train_pr, aes(sample = Parch)) + stat_qq()
p

train_pr$Parch <- log(train_pr$Parch+1)

g6 <- ggplot(data = train_pr, aes(x = train_pr$Fare)) + geom_histogram(bins = 20)
g6
g6_d <- ggplot(data = train_pr, aes(x = train_pr$Fare)) + geom_density(alpha = .2)
g6_d
p <- ggplot(train_pr, aes(sample = Fare)) + stat_qq()
p

train_pr$Fare <- log(train_pr$Fare+1)
test_pr$Fare <- log(test_pr$Fare+1)



#correlation
nums <- unlist(lapply(train_pr, is.numeric))  
df <- train_pr[, nums]
df_cor <- melt(cor(df))
head(df_cor)
g <- ggplot(data = df_cor, aes(x=Var1, y=Var2, fill=value)) + 
  geom_raster()
g <- g + theme(axis.text.x = element_text(angle = 90, hjust = 1))
g
df_cor[(df_cor$value > 0.7) | (df_cor$value < -0.7),] %>% na.omit()

##missing value imputing
train_pr <- missForest(train_pr)
train_pr$ximp
train_pr$OOBerror

test_pr <- missForest(test_pr)
test_pr$ximp
test_pr$OOBerror

#Combine dataset
train_set <- cbind(train$Survived, train_pr$ximp) %>% as.data.frame()
names(train_set)[1] <- "Survived"
head(train_set)


test.imp <- data.frame(test_pr$ximp)
summary(test.imp)
nrow(test.imp)
ncol(test.imp)



##modeling
#*************************************************************************
logis <- glm(Survived ~ ., data = train_set, family = "binomial")
summary(logis)

p <- predict(logis, train_set, type="response")
pr <- prediction(p, train_set$Survived)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
abline(a=0, b= 1)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

test_pr_linear <- predict(logis, newdata = test.imp, type="response")
summary(test_pr_linear)

plot(test_pr_linear)
test_pr_linear <- ifelse(test_pr_linear <= .5, 0,1)

test <- read.csv("test.csv",header = T, na.strings = c(" ",""))
result_1 <- cbind(test$PassengerId, test_pr_linear) %>% as.data.frame()
names(result_1)[1] <- names(test)[1]
names(result_1)[2] <- names(train)[1]
head(result_1)
write.csv(result_1, "result_1.csv")

#******************************************************************


#KAGGLE 50%
m_rf <- randomForest(Survived ~., data = train_set, importance = TRUE)
m_rf
rf <- predict(m_rf, newdata = test.imp, predict.all = TRUE)
summary(rf)
plot(rf$aggregate)
rf_pr <- as.data.frame(cbind(test$PassengerId, rf$aggregate))
head(rf_pr)
names(rf_pr)[1] <- names(test)[1]
names(rf_pr)[2] <- names(train)[1]
head(rf_pr)
rf_pr$Survived <- ifelse(rf_pr$Survived >= .5, 1 ,0)
write.csv(rf_pr, "result_2.csv")

#******************************************************************


random_index <- sample(1:nrow(train_set), nrow(train_set))
random_ames_train <- train_set[random_index, ]
# create hyperparameter grid
hyper_grid <- expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1, 3, 5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid)
#81
# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # train model
  gbm.tune <- gbm(
    formula = Survived ~ .,
    distribution = "bernoulli",
    data = random_ames_train,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)



gbm.fit <- gbm(
  formula = Survived ~ .,
  distribution = "bernoulli",
  data = train_set,
  bag.fraction = 0.65,
  n.minobsinnode = 5,
  n.trees = 29,
  interaction.depth = 3,
  shrinkage = 0.3,
  cv.folds = 10,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  
gbm1 <- predict(gbm.fit, newdata = test.imp, n.trees=29, type = "response")
gbm_pr1 <- as.data.frame(cbind(test$PassengerId, gbm1))
head(gbm_pr1)
names(gbm_pr1)[1] <- names(test)[1]
names(gbm_pr1)[2] <- names(train)[1]
gbm_pr1$Survived <- ifelse(gbm_pr1$Survived >= .5, 1 ,0)
write.csv(gbm_pr1,"result_3.csv") #0.13609
