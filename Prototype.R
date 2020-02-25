library(ggplot2)
library(glmnet)
library(pcLasso)
library(PRROC)


####################################################
# Partition a held-out set
####################################################
set.seed(1234)
pwd      <- dirname(rstudioapi::getSourceEditorContext()$path)
PATH     <- file.path(pwd, "data/loan_train.csv")
df_train <- read.csv(PATH)

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
pairs(~., data=train[, 1:10]) # pairwise scatterplot

####################################################
# Try a model
####################################################
pclasso  <- pcLasso(X, y, rat = .5, family = "binomial")
lasso    <- glmnet(X, y, alpha = 1, family = "binomial")
ridge    <- glmnet(X, y, alpha = 0, family = "binomial")
elastic  <- glmnet(X, y, alpha = .02, family = "binomial")

# Prevalence of class 1 is 1/3, so let's look at AUCPR instead of AUCROC
aucpr <- function(x) pr.curve(scores.class0 = x[yte == 0], scores.class1 = x[yte == 1])$auc.integral
auc.lasso   <- apply(predict(lasso,   xte, type='response'), 2, aucpr)
auc.pclasso <- apply(predict(pclasso, xte, type='response'), 2, aucpr)
auc.ridge   <- apply(predict(ridge,   xte, type='response'), 2, aucpr)
auc.elastic <- apply(predict(elastic, xte, type='response'), 2, aucpr)

l2norm <- function(x) sqrt( x %*% x)
forplot <- rbind(
  data.frame(x = apply(pclasso$beta, 2, l2norm), y=auc.pclasso, color="pcLasso"),
  data.frame(x = apply(lasso$beta,   2, l2norm), y=auc.lasso,   color="lasso"),
  data.frame(x = apply(ridge$beta,   2, l2norm), y=auc.ridge,   color="ridge"),
  data.frame(x = apply(elastic$beta, 2, l2norm), y=auc.elastic, color="elastic-net")
)

ggplot(forplot[forplot$x > 0, ], aes(x=x, y=y, color=color)) + 
  geom_point(size=1, alpha=.8) +
  geom_line(alpha=.5) +
  labs(x = "L2 norm of beta", y = "AUCPR", title = "Loan data", subtitle = "(held-out data)") 
