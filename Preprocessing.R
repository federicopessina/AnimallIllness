##########################################################################

# data preparation
library(tidyverse)
library(xgboost)

dataset_path = file.choose() # return path string
dataset_raw = read.csv2(dataset_path, header = TRUE, sep = ",")

names(dataset_raw)
summary(as.data.frame(dataset_raw))

## remove not informative fields
dataset_clean = dataset_raw %>%
  select(-Id) %>%                     # id is not useful
  select(-c(longitude, latitude))     # redundant information

# retrieve just numeric fields
dataset_numeric = dataset_clean %>% 
  select_if(is.numeric)               # select remaining numeric columns


##########################################################################

# explorative analysis

## relations between animal-fields
pairs(dataset_numeric[,1:5], pch = 19,
      lower.panel = NULL,
      na.action = na.omit)

## relations between human-fields 
pairs(dataset_numeric[,6:8], pch = 19,
      lower.panel = NULL,
      na.action = na.omit)

##########################################################################

## linear regression and significance test

linear_model.sumAtRisksumDestroyed <- lm(formula = sumAtRisk ~ sumDestroyed, 
                                         data = dataset_numeric,
                                         na.action = na.omit)
summary(linear_model.sumAtRisksumDestroyed)


##########################################################################

# pca

library(tidyverse)

## remove not informative fields
dataset_pca = dataset_raw %>%
  select(-Id) %>%  
  select(-c(longitude, latitude))     # redundant information


dataset_pca = dataset_pca %>% 
  select_if(is.numeric)  

dataset_pca = dataset_pca %>% select(-starts_with("human"))


## calculate principal components
results = prcomp(x = na.omit(dataset_pca), # data frame for pca
                 scale = TRUE,             # to have mean = 0 and SD = 1 before calculating
                 )

## reverse the signs
results$rotation = -1*results$rotation

results$rotation

biplot(results, scale = 1)                 # plot biplot of pca

results$sdev^2 / sum(results$sdev^2)       #calculate total variance explained by each principal component

## calculate total variance explained by each principal component

var_explained = results$sdev^2 / sum(results$sdev^2)

## create scree plot
qplot(c(1:5), var_explained) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab("Variance Explained") +
  ggtitle("Scree Plot") +
  ylim(0, 1)


library(devtools)
#install_github("vqv/ggbiplot")
require(ggbiplot)

summary(results)


##########################################################################

# logistic regression
library(xgboost)

dataset_humanremoved = dataset_numeric %>% select(-starts_with("human"))

logistic_data = dataset_humanremoved

str(logistic_data)



## remove fields related to humans

disease_labels = dataset_numeric %>% 
  select(humansAffected) %>%  # columns with the humans affected
  is.na() %>%                 # check if NA value
  magrittr::not()             # switch  boolean values


head(disease_labels)                # target variable
head(dataset_clean$humansAffected)  # of the original column

head(dataset_clean$country)   # print example of country
unique(dataset_clean$country) # print every possible country in db

region = model.matrix(~country-1, dataset_clean) # convert factors into one-hot encoded variables
head(region)

head(dataset_clean$speciesDescription)  # print example


dataset_numeric$is_domestic = str_detect(dataset_clean$speciesDescription, "domestic") # retrieve just domestic species
head(dataset_numeric$is_domestic)       # print example 


### generate species list
species_list = dataset_clean$speciesDescription %>%
  str_replace("[[:punct:]]", "") %>%    # remove punctuation
  str_extract("[a-z]*$")                # extract the last word

species_list = tibble(species = species_list) # convert to df
head(species_list)                            # print example
options(na.action='na.pass')                  # maintain NAs
species = model.matrix(~species-1, species_list)
head(species)                                 # print example

### add our one-hot encoded variable and convert the dataframe into a matrix
dataset_numeric <- cbind(dataset_numeric, region, species)
dataset_numeric_matrix <- data.matrix(dataset_numeric)
head(dataset_numeric_matrix) # print example

## train-test split 70-30
training_lenght = round(length(disease_labels) * .70)

### training
train_data = dataset_numeric_matrix[1:training_lenght,]
train_labels = disease_labels[1:training_lenght]

### test
test_data = dataset_numeric_matrix[-(1:training_lenght),]
test_labels = disease_labels[-(1:training_lenght)]

### train and test data to data matrix
data_train = xgb.DMatrix(data = train_data, label = train_labels)
data_test = xgb.DMatrix(data = test_data, label = test_labels)


## model
set.seed(9876)

dataset_clean = dataset_clean[sample(1:nrow(dataset_clean)),]

model = xgboost(data = data_train,              # the data   
                 nround = 2,                    # maximum recursion
                 objective = "binary:logistic") # objective func

prediction = predict(model, data_test)

error = mean(as.numeric(prediction > 0.5) != test_labels)
print(paste("test-error = ", error))

# train a model using our training data
model_tuned = xgboost(data = data_train,  # the data           
                       max.depth = 3,     # the maximum depth of each decision tree
                       nround = 10,       # number of boosting rounds
                       early_stopping_rounds = 3,
                       objective = "binary:logistic", # the objective function
                       scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
                       gamma = 1)         # regularization term