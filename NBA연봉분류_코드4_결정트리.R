#------------------------------------------------------------------------------
## 데이터마이닝 프로젝트 최종 코딩 - 최현석, 황성진
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 코드4 - 결정트리 모델
#------------------------------------------------------------------------------
# 0. 데이터 전처리
#------------------------------------------------------------------------------
nba <- read.csv("C:\\Users\\chs02\\OneDrive\\바탕 화면\\데마프로젝트\\NBA_Player_Salaries_(2022-23).csv")

nba <- nba[, -1]      # ID 변수 제거
nba <- na.omit(nba)   # 결측치 제거
nba$Position <- sub("-.*", "", nba$Position) # 단일 포지션으로

# 연봉 분류 변수 생성
salary_cap = 123655000
nba$pct_cap <- (as.numeric(nba$Salary) / salary_cap) * 100

bins <- c(0, 2, 13, 100)
labels <- c("Low", "Mid", "High")
nba$sal_tier <- cut(nba$pct_cap, breaks = bins, labels = labels, right = FALSE)

#------------------------------------------------------------------------------
# 0.5. 데이터 분할
#------------------------------------------------------------------------------
selected.var <- c("Years.of.Service", "GS",
                  "FGA", "X2PA", "FTA", "AST", "TOV", "PPG", "VORP")

set.seed(1)
train.index <- sample(row.names(nba), 0.8 * nrow(nba))
valid.index <- setdiff(row.names(nba), train.index)

train.df <- nba[train.index, selected.var]
valid.df <- nba[valid.index, selected.var]

# 종속 변수 추가
train.df$sal_tier <- nba[train.index, "sal_tier"]
valid.df$sal_tier <- nba[valid.index, "sal_tier"]


#------------------------------------------------------------------------------
# 1. 나무모델 - 5-fold 반복 없음
#------------------------------------------------------------------------------
# (4-1-1). [가지치기 전]
## 1. 트리 모델 학습
set.seed(1)
library(rpart)
tree.model <- rpart::rpart(sal_tier ~ ., data = train.df, method = "class", 
                           cp = 0, xval=5)

## 2. 검증셋 예측 및 평가
tree.pred <- predict(tree.model, newdata = valid.df, type = "class")

caret::confusionMatrix(tree.pred, valid.df$sal_tier)

## 3. 완전 성장 나무 시각화화
library(rpart.plot)  
prp(tree.model,
    type = 1,         
    extra = 1,       
    faclen = 0,        
    varlen = -10,      
    cex = 0.8,        
    main = "Classification Tree for Salary Tier")

#------------------------------------------------------------------------------
# (4-1-2). [가지치기 후]
## 1. 트리 복잡도 출력 + 가지치기
printcp(tree.model)

## 2. 최적 가지치기
best_cp <- tree.model$cptable[which.min(tree.model$cptable[,"xerror"]), "CP"]
best_cp
pruned.ct <- prune(tree.model, cp = best_cp)

length(pruned.ct$frame$var[pruned.ct$frame$var == "<leaf>"]) # 끝 노드 수

## 3. 가지치기 모델 성능
tree.pred2 <- predict(pruned.ct, newdata = valid.df, type = "class")
caret::confusionMatrix(tree.pred2, valid.df$sal_tier)

## 4. 가지치기 나무 시각화
library(rpart.plot)  
prp(pruned.ct,
    type = 1,         
    extra = 1,       
    faclen = 0,        
    varlen = -10,      
    cex = 0.8,        
    main = "Classification Tree for Salary Tier")


#------------------------------------------------------------------------------
# 2. 나무모델 - 5-fold 5번 반복
#------------------------------------------------------------------------------
library(caret)
library(rpart)

## (4-2-1). 교차검증 설정
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

## (4-2-2). 분류 나무 학습
set.seed(1)
tree_model <- train(sal_tier ~ ., 
                    data = train.df, 
                    method = "rpart", 
                    trControl = ctrl,
                    tuneLength = 10)  # cp 값을 자동 탐색 (자동 가지치기)

## (4-2-3). 결과 확인
print(tree_model)
plot(tree_model)

tree_model$resample$Accuracy  # 25개의 정확도 추정치
summary(tree_model$resample$Accuracy)  # 평균 성능
sd(tree_model$resample$Accuracy)       # 표준편차

boxplot(tree_model$resample$Accuracy, main = "Classification Tree Cross-Validation Accuracy", ylab = "Accuracy")


## (4-2-4). 교차검증 tree_model의 예측 성능 평가
# 테스트셋 예측
tree.pred <- predict(tree_model, newdata = valid.df)

# 성능 평가
confusionMatrix(tree.pred, valid.df$sal_tier)


## (4-2-5). 최종 결정된 트리 모델 시각화
final_tree <- tree_model$finalModel

prp(final_tree,
    type = 1,         
    extra = 1,       
    faclen = 0,        
    varlen = -10,      
    cex = 0.8,        
    main = "Classification Tree for Salary Tier")

#------------------------------------------------------------------------------
# 3. ROC-AUC 분석 (One-vs-All)
#------------------------------------------------------------------------------
library(pROC)
library(ggplot2)

# (4-3-1). 테스트셋에 대한 예측 확률 추출
tree.prob <- predict(tree_model, newdata = valid.df, type = "prob")
true_labels <- valid.df$sal_tier

# (4-3-2). ROC 곡선 시각화 과정
roc_list <- list()
auc_labels <- c()

for (class in levels(true_labels)) {
  actual_binary <- ifelse(true_labels == class, 1, 0)
  prob <- tree.prob[[class]]
  
  roc_obj <- roc(actual_binary, prob, quiet = TRUE)
  roc_list[[class]] <- roc_obj
  
  auc_val <- auc(roc_obj)
  auc_labels <- c(auc_labels, paste0(class, " (AUC = ", round(auc_val, 3), ")"))
  cat(paste("AUC for", class, ":", round(auc_val, 4)), "\n")
}

# ROC 곡선용 데이터프레임 생성
roc_df <- do.call(rbind, lapply(seq_along(roc_list), function(i) {
  roc_curve <- roc_list[[i]]
  data.frame(
    FPR = 1 - rev(roc_curve$specificities),
    TPR = rev(roc_curve$sensitivities),
    Class = auc_labels[i]
  )
}))

# ROC 곡선 시각화 (한 화면에)
ggplot(roc_df, aes(x = FPR, y = TPR, color = Class)) +
  geom_line(linewidth = 1.3) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(title = "One-vs-All ROC Curves (Decision Tree)",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal() +
  theme(theme(
    legend.title = element_blank(),
    text = element_text(size = 14),
    legend.text = element_text(size = 12)
  ))

#------------------------------------------------------------------------------
# 4. 변수 중요도 시각화
#------------------------------------------------------------------------------
library(tidyverse)

importance <- varImp(tree_model)

# (5-4-1). 상위 변수 10개 가져오기
importance_df <- as.data.frame(importance$importance)
importance_df$Variable <- rownames(importance_df)
importance_df <- importance_df %>%
  arrange(desc(Overall)) %>%
  slice(1:10)

# 변수 순서 고정
importance_df$Variable <- factor(importance_df$Variable, levels = rev(importance_df$Variable))

# (5-4-2). 시각화
ggplot(importance_df, aes(x = Variable, y = Overall, fill = Overall)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = round(Overall, 1)), hjust = -0.2, size = 4.2) +
  scale_fill_gradient(low = "#D6F0FF", high = "#5DADE2") +
  coord_flip() +
  labs(
    title = " 변수 중요도 (의사결정나무 기준)",
    x = NULL,
    y = "Importance Score"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 16, face = "bold")
  ) +
  ylim(0, max(importance_df$Overall) * 1.15)

#------------------------------------------------------------------------------

