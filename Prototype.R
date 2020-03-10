library(ggplot2)
library(glmnet)
library(PRROC)
library(splines)

####################################################
# Partition a held-out set
####################################################
set.seed(1234)
#pwd      <- dirname(rstudioapi::getSourceEditorContext()$path)
#PATH     <- file.path(pwd, "data/loan_train.csv")
df_train <- read.csv("data/loan_train.csv")

# Do a stratified split to have some held-out data
ones  <- df_train[df_train[, "default"] == 1, ]
zeros <- df_train[df_train[, "default"] == 0, ]

ones.sample  <- sample(1:nrow(ones),  .1 * nrow(ones))
zeros.sample <- sample(1:nrow(zeros), .1 * nrow(zeros))

heldout <- rbind(ones[ones.sample, ], zeros[zeros.sample, ])
train   <- rbind(ones[!(1:nrow(ones) %in% ones.sample), ], 
                 zeros[!(1:nrow(zeros) %in% zeros.sample), ]
            )
train   <- train[sample(1:nrow(train)), ]  # shuffle the rows

####################################################
# Helper functions for processing data
####################################################
#transform employment to a quantitative variable
process_employment <- function(employment_str) {
  if (is.na(employment_str)) {
    employment_val <- NA
  }
  else if (employment_str == '10+') {
    employment_val <- 10
  }
  else if (employment_str == '< 1') {
    employment_val <- 0
  }
  else {
    employment_val <- as.numeric(employment_str)
  }
}

#transform quality to a quantitative variable
transform_quality_score <- function(quality_score_str) {
  return(as.numeric(substr(quality_score_str, 2, 2)))
}

train$employment <- sapply(train$employment, process_employment)
#impute employment NAs with mean
train$employment[is.na(train$employment)] <- mean(train$employment, na.rm = TRUE)
train$quality <- sapply(train$quality, transform_quality_score)

heldout$employment <- sapply(heldout$employment, process_employment)
#impute employment NAs with mean
heldout$employment[is.na(heldout$employment)] <- mean(heldout$employment, na.rm = TRUE)
heldout$quality <- sapply(heldout$quality, transform_quality_score)

# Convert to matrices and deal with dummy variables:
X   <- makeX(train[, 2:ncol(train)], heldout[, 2:ncol(heldout)]) 
xte <- X$xtest; X <- X$x
yte <- heldout[, 1]; y <- train[, 1]

# Just ignore the "employment" column for now
X[is.na(X)] <- 0
xte[is.na(xte)] <- 0

####################################################
# EDA
####################################################
#pairs(~., data=train[, 1:10]) # pairwise scatterplot

###################################################
# Splines
###################################################

cv.spline <- function(df){
	Ntrain <- ns(train$inc,   df = df, intercept = TRUE)
	Ntest  <- ns(heldout$inc, df = df, intercept = TRUE)
	colnames(Ntrain) <- paste0("Ninc", 1:df)
	colnames(Ntest)  <- paste0("Ninc", 1:df)
	model <- glm(as.formula(paste("default ~ . + ", "(",
                             paste0(rep("Ninc", df), 1:df, collapse = " + "),
                             ")")),
	  	   family = "binomial",
            	 data = data.frame(train, Ntrain))

	heldout <- data.frame(heldout, Ntest)
	pred  <- predict(model, heldout, type="response") > .5
	accu  <- mean(pred == heldout$default)
	accu
}
cvs <- sapply(2:22, cv.spline)
bestspline.df <- 1 + which.max(cvs)
