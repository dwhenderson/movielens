##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##########################################################

# Begin own code

library(lubridate)
library(stringr)

# Cleaning the datasets
# Add columns for title year, rating year, and boolean variables for genres

edx <- edx %>% mutate(
  rating_year = year(as_datetime(timestamp)),
  title_year = str_extract(title, "\\(\\d+\\)$"),
  title_year = as.numeric(str_remove_all(title_year, "\\(|\\)")),
  comedy = ifelse(str_detect(genres, "Comedy"), 1, 0),
  action = ifelse(str_detect(genres, "Action"), 1, 0),
  children = ifelse(str_detect(genres, "Children"), 1, 0),
  adventure = ifelse(str_detect(genres, "Adventure"), 1, 0),
  animation = ifelse(str_detect(genres, "Animation"), 1, 0),
  drama = ifelse(str_detect(genres, "Drama"), 1, 0),
  crime = ifelse(str_detect(genres, "Crime"), 1, 0),
  scifi = ifelse(str_detect(genres, "Sci-Fi"), 1, 0),
  horror = ifelse(str_detect(genres, "Horror"), 1, 0),
  thriller = ifelse(str_detect(genres, "Thriller"), 1, 0),
  mystery = ifelse(str_detect(genres, "Mystery"), 1, 0),
  romance = ifelse(str_detect(genres, "Romance"), 1, 0),
  fantasy = ifelse(str_detect(genres, "Fantasy"), 1, 0),
  musical = ifelse(str_detect(genres, "Musical"), 1, 0),
  war = ifelse(str_detect(genres, "War"), 1, 0),) %>%
  select(-genres, -timestamp)


validation <- validation %>% mutate(
  rating_year = year(as_datetime(timestamp)),
  title_year = str_extract(title, "\\(\\d+\\)$"),
  title_year = as.numeric(str_remove_all(title_year, "\\(|\\)")),
  comedy = ifelse(str_detect(genres, "Comedy"), 1, 0),
  action = ifelse(str_detect(genres, "Action"), 1, 0),
  children = ifelse(str_detect(genres, "Children"), 1, 0),
  adventure = ifelse(str_detect(genres, "Adventure"), 1, 0),
  animation = ifelse(str_detect(genres, "Animation"), 1, 0),
  drama = ifelse(str_detect(genres, "Drama"), 1, 0),
  crime = ifelse(str_detect(genres, "Crime"), 1, 0),
  scifi = ifelse(str_detect(genres, "Sci-Fi"), 1, 0),
  horror = ifelse(str_detect(genres, "Horror"), 1, 0),
  thriller = ifelse(str_detect(genres, "Thriller"), 1, 0),
  mystery = ifelse(str_detect(genres, "Mystery"), 1, 0),
  romance = ifelse(str_detect(genres, "Romance"), 1, 0),
  fantasy = ifelse(str_detect(genres, "Fantasy"), 1, 0),
  musical = ifelse(str_detect(genres, "Musical"), 1, 0),
  war = ifelse(str_detect(genres, "War"), 1, 0),) %>%
  select(-genres, -timestamp)


# Split edx into training and test sets

test_ind <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)

edx_train <- edx[-test_ind,]

temp <- edx[test_ind,]

edx_test <- temp %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

rm(temp)


# Estimate conditional means for dramas and non-dramas

mu_drama <- edx_train %>%
  group_by(drama) %>%
  summarize(avg_rating = mean(rating)) %>%
  slice(2) %>%
  pull(avg_rating)

mu_not_drama <- edx_train %>%
  group_by(drama) %>%
  summarize(avg_rating = mean(rating)) %>%
  slice(1) %>%
  pull(avg_rating)

edx_train <- edx_train %>%
  mutate(mu = case_when(drama == 1 ~ mu_drama,
                        drama == 0 ~ mu_not_drama))

edx_test <- edx_test %>%
  mutate(mu = case_when(drama == 1 ~ mu_drama,
                        drama == 0 ~ mu_not_drama))

# Regularized movie, user, and year effects
# Find lambda that minimizes RMSE of predictions

lambdas <- c(0, seq(3.5, 6, 0.25))

rmses <- sapply(lambdas, function(l){
  
  b_yr <- edx_train %>%
    group_by(title_year) %>%
    summarize(b_yr = sum(rating - mu)/(n() + l))
  
  b_i <- edx_train %>%
    left_join(b_yr, by = "title_year") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_yr)/(n() + l))
  
  b_u <- edx_train %>%
    left_join(b_yr, by = "title_year") %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_yr - b_i)/(n() + l))
  
  predictions <- edx_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_yr, by = "title_year") %>%
    mutate(pred = mu + b_i + b_u + b_yr) %>%
    pull(pred)
  
  return(RMSE(edx_test$rating, predictions))
  
})

rm(mu_drama, mu_not_drama)
edx_train <- edx_train %>% select(-mu)
edx_test <- edx_test %>% select(-mu)
best_lambda <- lambdas[which.min(rmses)]
min(rmses)


# Train model on full edx dataset using optimized lambda

mu_drama <- edx %>%
  group_by(drama) %>%
  summarize(avg_rating = mean(rating)) %>%
  slice(2) %>%
  pull(avg_rating)

mu_not_drama <- edx %>%
  group_by(drama) %>%
  summarize(avg_rating = mean(rating)) %>%
  slice(1) %>%
  pull(avg_rating)

edx <- edx %>%
  mutate(mu = case_when(drama == 1 ~ mu_drama,
                        drama == 0 ~ mu_not_drama))

validation <- validation %>%
  mutate(mu = case_when(drama == 1 ~ mu_drama,
                        drama == 0 ~ mu_not_drama))

b_yr <- edx %>%
  group_by(title_year) %>%
  summarize(b_yr = sum(rating - mu)/(n() + best_lambda))

b_i <- edx %>%
  left_join(b_yr, by = "title_year") %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu - b_yr)/(n() + best_lambda))

b_u <- edx %>%
  left_join(b_yr, by = "title_year") %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_yr - b_i)/(n() + best_lambda))


# Use trained model to make predictions in final holdout test set

predictions <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_yr, by = "title_year") %>%
  mutate(pred = mu + b_i + b_u + b_yr) %>%
  pull(pred)

# RMSE of final predictions < 0.86490

RMSE(validation$rating, predictions) 


# Remove leftover objects and columns

rm(b_yr, b_i, b_u, mu_drama, mu_not_drama, lambdas, best_lambda, rmses)
edx <- edx %>% select(-mu)
validation <- validation %>% select(-mu)
