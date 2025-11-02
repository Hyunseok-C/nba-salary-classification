#------------------------------------------------------------------------------
## 데이터마이닝 프로젝트 최종 코딩 - 최현석, 황성진
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 코드3 - 나이브베이즈 모델
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
# 0.5. 수치형 변수의 범주형 변환과 데이터 분할
#------------------------------------------------------------------------------
# (3-0.5-1) 수치형 변수의 범주형 변환
selected_vars <- c("Years.of.Service", "GS",
                  "FGA", "X2PA", "FTA", "AST", "TOV", "PPG", "VORP")

library(e1071)
library(dplyr)

# 연속형 변수만 필터링
num_vars <- selected_vars[sapply(nba[selected_vars], is.numeric)]

# 전처리 시작
df_trans <- nba %>% select(all_of(selected_vars), sal_tier)


# 왜도 기준 분기 → cut 결과를 factor로 저장하고, 구간 레이블도 명시
skew_summary <- data.frame(Variable = character(), Skewness = numeric(), Binning = character())
for (var in num_vars) {
  x <- nba[[var]]
  sk <- skewness(x, na.rm = TRUE)
  
  if (abs(sk) >= 1) {
    # 왜도가 큰 경우 → 사분위수 기준
    df_trans[[var]] <- cut(x,
                           breaks = quantile(x, probs = seq(0, 1, 0.25), na.rm = TRUE),
                           include.lowest = TRUE,
                           labels = c("Q1", "Q2", "Q3", "Q4"),
                           ordered_result = TRUE)
    message(paste(var, ": 사분위수 기준 범주화 (왜도 =", round(sk, 2), ")"))
    bin_type <- "Quantile"
  } else {
    # 왜도가 작을 경우 → 등간격 기준
    df_trans[[var]] <- cut(x,
                           breaks = 3,
                           include.lowest = TRUE,
                           labels = c("Low", "Mid", "High"),
                           ordered_result = TRUE)
    message(paste(var, ": 등간격 범주화 (왜도 =", round(sk, 2), ")"))
    bin_type <- "EqualWidth"
  }
  skew_summary <- rbind(skew_summary,
                        data.frame(Variable = var,
                                   Skewness = round(sk, 2),
                                   Binning = bin_type))
}

# 왜도별 범주형 변환 방식 시각화
ggplot(skew_summary, aes(x = reorder(Variable, Skewness), y = Skewness, fill = Binning)) +
  geom_col(width = 0.7, color = "black") +
  coord_flip() +
  geom_text(aes(label = round(Skewness, 2)), hjust = -0.1, size = 4) +
  scale_fill_manual(values = c("Quantile" = "skyblue", "EqualWidth" = "salmon")) +
  labs(title = "Skewness and Binning Method per Variable",
       x = "Variable", y = "Skewness") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")


#------------------------------------------------------------------------------
# (3-0.5-2) 데이터 분할
library(caret)
set.seed(1)
train_idx <- createDataPartition(df_trans$sal_tier, p = 0.8, list = FALSE)
train_data <- df_trans[train_idx, ]
test_data <- df_trans[-train_idx, ]

train.index <- sample(row.names(nba), 0.8 * nrow(nba))
valid.index <- setdiff(row.names(nba), train.index)

#------------------------------------------------------------------------------
# 1. 나이브베이즈 모델 - 교차검증 5-fold x 5번 반복
#------------------------------------------------------------------------------
# (3-1-1). 교차검증 설정 (5-fold ×  5)
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

# (3-1-2). 나이브 베이즈 모델 학습
set.seed(1)
nb_model <- train(sal_tier ~ ., 
                  data = train_data,
                  method = "naive_bayes",
                  trControl = ctrl)
# 교차검증 결과 출력
print(nb_model)

# (3-1-3). 교차검증 정확도 요약 통계
summary(nb_model$resample$Accuracy) # 정확도
summary(nb_model$resample$Kappa)    # kappa
sd(nb_model$resample$Accuracy)      # 정확도 표준편차
boxplot(nb_model$resample$Accuracy, main = "Naive Bayes Cross-Validation Accuracy", ylab = "Accuracy")

# (3-1-4). 검증셋 예측
pred <- predict(nb_model, newdata = test_data)

# (3-1-5). 성능 평가
confusionMatrix(pred, test_data$sal_tier)

#------------------------------------------------------------------------------
# 2. ROC-AUC 분석 (One-vs-All)
#------------------------------------------------------------------------------
## [시각화 - 3개 클래스 ROC 곡선 합치기]
library(pROC)

# (3-2-1). 테스트셋에 대한 예측 확률 추출
pred_prob <- predict(nb_model, newdata = test_data, type = "prob")
true_labels <- test_data$sal_tier


# (3-2-2). ROC 곡선 시각화 과정
# ROC curve 객체 저장
roc_list <- list()
auc_labels <- c()

for (class in levels(true_labels)) {
  actual_binary <- ifelse(true_labels == class, 1, 0)
  prob <- pred_prob[[class]]
  
  roc_obj <- roc(actual_binary, prob)
  roc_list[[class]] <- roc_obj
  
  auc_val <- auc(roc_obj)
  auc_labels <- c(auc_labels, paste0(class, " (AUC = ", round(auc_val, 3), ")"))
}

# ggplot용 데이터프레임 변환
roc_df <- do.call(rbind, lapply(seq_along(roc_list), function(i) {
  roc_curve <- roc_list[[i]]
  data.frame(
    Specificity = rev(roc_curve$specificities),
    Sensitivity = rev(roc_curve$sensitivities),
    Class = auc_labels[i]  # AUC 수치 포함
  )
}))

# ggplot2 시각화 - 클래스 3개 ROC
ggplot(roc_df, aes(x = 1 - Specificity, y = Sensitivity, color = Class)) +
  geom_line(linewidth = 1.3) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(title = "One-vs-All ROC Curves (Naive Bayes)",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal() +
  theme(theme(
    legend.title = element_blank(),
    text = element_text(size = 14),       
    legend.text = element_text(size = 12) 
  ))

#------------------------------------------------------------------------------

