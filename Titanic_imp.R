library(ggplot2)
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

test$Survived <- NA
total <- rbind(train,test)
sapply(total, function(x) {sum(is.na(x))})

total$Pclass <- as.ordered(total$Pclass)
total$Sex <- as.factor(total$Sex)
total$Embarked <- as.factor(total$Embarked)
total$Name <- as.character(total$Name)

#*********************************************************
#Variable comnbination: SEX & PClass
##Sig: P_sex
total$P_sex[total$Pclass == "1" & total$Sex == "female"] <- "F1"
total$P_sex[total$Pclass == "2" & total$Sex == "female"] <- "F2"
total$P_sex[total$Pclass == "3" & total$Sex == "female"] <- "F3"
total$P_sex[total$Pclass == "1" & total$Sex == "male"] <- "M1"
total$P_sex[total$Pclass == "2" & total$Sex == "male"] <- "M2"
total$P_sex[total$Pclass == "3" & total$Sex == "male"] <- "M3"


#*********************************************************
#Feature engineering
#1 title generation: sig variable title
total$title <- sapply(total$Name, function(x) {strsplit(x, split = '[,.]')[[1]][2]})
total$title <- sub(' ', '', total$title)
num_title <- table(total$title) %>% as.data.frame()

total$title[total$title %in% c("Lady", "Mlle")] <- "Ms"
total$title[total$title %in% c("Sir")] <- "Mr"
total$title[!(total$title %in% c("Master", "Miss", "Mr", "Mrs"))] <- "Other"
total$title <- as.factor(total$title)
table(total$title)

g1 <- ggplot(total, aes(total$title, total$Survived)) + geom_jitter(color = "sky blue")
g1

ggplot(total[!is.na(total$Survived),], aes(x = title, fill = factor(Survived))) + 
  geom_bar(stat = "count", position = "stack") + 
  labs(x = "title") + theme_grey()


#2 family number creating
total$family_name <- sapply(total$Name, function(x) {strsplit(x, split = '[,.]')[[1]][1]})
total$family_num <- total$SibSp + total$Parch + 1
g1 <- ggplot(total, aes(total$family_num, total$Survived)) + geom_jitter(color = "sky blue")
g1

total$familyname_num <- paste0(total$family_name,total$family_num)
head(total$familyname_num)

ggplot(total[!is.na(total$Survived),], aes(x = family_num, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') + theme_grey()

inc <- table(total$familyname_num) %>% as.data.frame()
inc$Var1 <- as.character(inc$Var1)
substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}
inc$num <- substrRight(inc$Var1, 1)
inconsis <- inc[inc$num != inc$Freq, ]
nrow(inconsis)
head(inconsis)
#freq: last name adding together, num:family size

#dataquality problem: travel appears to be single but not
total$Ticket_group <- sub("..$", "", total$Ticket)
names(total)
others <- total %>% 
  select(Pclass, Name, SibSp, Parch, family_name, family_num, Ticket_group) %>%
  filter(family_num == 1) %>%
  group_by(Ticket_group, family_name) %>%
  summarise(count=n())
head(others)
rest <- others[others$count>1,] %>% as.data.frame()
rest
names(total)

others2 <- total[(total$Ticket_group %in% rest$Ticket_group & total$family_name %in% rest$family_name 
       & total$family_num == 1 ),
    c('PassengerId', 'family_name', 'title', 'Age', 
      'Ticket', 'Ticket_group', 'family_num', 'SibSp', 'Parch')]
others2 <- left_join(others2, rest, by = c("family_name", "Ticket_group"))
head(others2)
head(total)

total <- left_join(total, others2, 
                   by = c('PassengerId',"family_name", "Ticket_group", 
                          'title', 'Age', 'family_num', 'SibSp', 'Parch', 'Ticket'))
summary(total$count)
for (i in 1:nrow(total)){
  if (!is.na(total$count[i])){
    total$family_num[i] <- total$count[i]
  }
}
total$family_num %>% summary()

#Single traveler

total$issingle <- ifelse(total$family_num == 1, 1, 0)

#Are they buy tickect with one group or not?

Ticket_G <- total %>%
  select(Ticket) %>%
  group_by(Ticket) %>%
  summarise(count=n())
names(Ticket_G)[2] <- "Tgroup"
total <- left_join(total,Ticket_G)
total <- total[, -which(names(total) %in% c("count"))]

head(total)
total$Tgroup <- as.numeric(total$Tgroup)

ggplot(data = total[!is.na(total$Survived), ]) + aes(x = Tgroup, fill = factor(Survived)) + 
  geom_bar(stat='count', position='dodge') + 
  theme_gray()
## Group ticketing info: sig variable -- Group
total$Group[total$Tgroup > 4] <- "Large" 
total$Group[total$Tgroup == 1] <- "Single"
total$Group[total$Tgroup == 2] <- "Double"
total$Group[total$Tgroup >=3 & total$Tgroup <= 4 ] <- "small"

head(total)

ggplot(total[!is.na(total$Survived),], aes(x = Group, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  labs(x = 'Final Group Categories') + theme_grey()
## whether fare and p_class has a relationship
lm(total$Fare ~ total$Pclass) %>% summary() # sig
summary(total$Pclass)
lm(total$Fare ~ total$Embarked) %>% summary() # sig

total %>%
  select(Fare, Embarked, Pclass) %>%
  group_by(Embarked, Pclass) %>%
  summarise(Fee = mean(Fare)) # it seems the prce is no right

## is group ticketing matters with fare?

lm(total$Fare ~ total$Group) %>% summary()
lm(total$Tgroup ~ total$Pclass) %>% summary()
total$fee <- total$Fare / total$Tgroup
summary(total$fee)

total %>%
  select(fee, Embarked, Pclass) %>%
  group_by(Embarked, Pclass) %>%
  summarise(Fee = median(fee))
# the result seems not so reasonable, for Q with class 2, 3

##****missing value imputing
#Embarked
is.na(total$Embarked) %>% summary()
total[is.na(total$Embarked),]
total$Embarked[total$PassengerId == 62 | total$PassengerId == 830] <- "C" 

#Fee
is.na(total$fee) %>% summary()
total[is.na(total$fee),]
total$fee[total$PassengerId == 1044] <- 7.8

#check fee variable

ggplot(total[!is.na(total$Survived), ]) + aes(x = fee, fill = "skyblue") + 
  geom_histogram(binwidth = 0.01, fill='skyblue') + 
  theme_grey()

total$fee <- log(total$fee + 1)
#******************************************************************************
#checking other missing values

ggplot(total[!is.na(total$Survived), ]) + aes(x = Age, fill = "skyblue") + 
  geom_histogram(binwidth = 1, fill='skyblue') + 
  theme_grey()

ggplot(total[!is.na(total$Survived), ]) + aes(x = title, y = Age, fill = factor(Pclass)) + 
  geom_boxplot() + theme_gray()

Age_info <- total[!is.na(total$Age), ] %>% 
  select(Age, Pclass, title) %>%
  group_by(title, Pclass) %>%
  summarise(avg_Age = mean(Age))
# this might work, but can we use regression to predict the missing value?

Age_pr <- lm(Age ~ Pclass + Sex + SibSp + Embarked + title + Group, 
             data = total[!is.na(total$Age),])
summary(Age_pr)

Age_imp <- predict(Age_pr, newdata = total) 
total$age_imp <- predict(Age_pr, newdata = total) 
Age_imp %>% as.data.frame()
total$Age[is.na(total$Age)] <- total$age_imp[is.na(total$Age)]

is.na(total$Age) %>% summary()
ggplot(total[!is.na(total$Survived), ]) + aes(x = Age, fill = factor(Survived)) + 
  geom_histogram(binwidth = 1) + 
  theme_grey()

#since iskid can be a major variable to predict, creat it

total$iskid <- ifelse(total$Age < 15, 1, 0)
summary(total$iskid)

###others?? only Cabin has missing values
sapply(total[!is.na(total$Survived),], function(x) sum(is.na(x)))
summary(total$Cabin)
head(total$Cabin)
total$Cabin <- as.character(total$Cabin)
total$Cabin[is.na(total$Cabin)] <- "N"
total$Cabin <- substring(total$Cabin, 1, 1)
ggplot(total[(!is.na(total$Survived)& total$Cabin!='N'),], aes(x = Cabin, fill = factor(Survived))) +
  geom_bar(stat='count') + theme_grey() + 
  facet_grid(.~Pclass) + labs(title="Survivor split by class and Cabin")

ggplot(total[!is.na(total$Survived),], aes(x = Cabin, fill = factor(Survived))) +
  geom_bar(stat='count') + theme_grey() + 
  facet_grid(.~Pclass) + labs(title="Survivor split by class and Cabin")

##Embark?
p1 <- ggplot(total[!is.na(total$Survived),]) + aes(x = Embarked, fill = factor(Survived)) +
  geom_bar(stat = "count") + theme_light()
p2 <- ggplot(total[!is.na(total$Survived),]) + aes(x = Embarked, fill = factor(Survived)) +
  geom_bar(stat = "count", position = 'fill') + theme_light()

grid.arrange(p1, p2, nrow = 1)


#**********************************************************************************
#modeling
###dataset
train_f <- total[!is.na(total$Survived),]
test_f <- total[is.na(total$Survived),]
write.csv(train_f, "train_f.csv")
write.csv(test_f, "test_f.csv")

library(h2o)
h2o.init(nthreads = -1)
h2o.removeAll()


args(h2o.deeplearning)

projdata <- "C:\\Users\\saily\\Desktop\\000.Titanic\\train_f.csv"
testdata <- "C:\\Users\\saily\\Desktop\\000.Titanic\\test_f.csv"
rawdata <- h2o.importFile(projdata)
test_f <- h2o.importFile(testdata)
head(rawdata)
head(test_f)
rawdata$Survived <- as.factor(rawdata$Survived) 
h2o.levels(rawdata$Survived) 

splits <- h2o.splitFrame(data = rawdata, 
                         ratios = c(0.7, 0.15)) 
train <- splits[[1]]
valid <- splits[[2]]
test <- splits[[3]]

nrow(train) #33271
nrow(valid) #7145
nrow(test)  #7147

names(rawdata)
y <- "Survived"
x <- setdiff(names(rawdata), 
             c(y,"PassengerId", "Name",
             "SibSp", "Parch","Ticket","Tgroup", "Fare", "Embarked", 
             "family_name", "familyname_num","Ticket_group", "age_imp"))
print(x)



gbm_params1 <- list(learn_rate = c(0.01, 0.1),
                    max_depth = c(3, 5, 9),
                    sample_rate = c(0.8, 1.0),
                    col_sample_rate = c(0.2, 0.5, 1.0))

# Train and validate a cartesian grid of GBMs
gbm_grid1 <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid1",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 100,
                      hyper_params = gbm_params1)

# Get the grid results, sorted by validation AUC
gbm_gridperf1 <- h2o.getGrid(grid_id = "gbm_grid1", 
                             sort_by = "auc", 
                             decreasing = TRUE)
print(gbm_gridperf1)

#increase trees
gbm_grid2 <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid2",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 200,
                      hyper_params = gbm_params1)

# Get the grid results, sorted by validation AUC
gbm_gridperf2 <- h2o.getGrid(grid_id = "gbm_grid2", 
                             sort_by = "auc", 
                             decreasing = TRUE)
print(gbm_gridperf2)

best_gbm1 <- h2o.getModel(gbm_gridperf2@model_ids[[1]])
best_gbm_perf1 <- h2o.performance(model = best_gbm1,
                                  newdata = test)
result1 <- h2o.predict(best_gbm1, newdata = test_f)
result1 <- h2o.cbind(test_f$PassengerId, result1)
names(result1)[2] <- "Survived"
result1 <- result1[, 1:2]
h2o.exportFile(result1, "C:\\Users\\saily\\Desktop\\000.Titanic\\result_h2o1.csv")

h2o.shutdown()
Y
