#------------------------------------------------------------------------------
## 데이터마이닝 프로젝트 최종 코딩 - 최현석, 황성진
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 코드5 - 랜덤포레스트 모델
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
# 1. 랜덤포레스트
#------------------------------------------------------------------------------
library(randomForest)
library(caret)

rf_model <- randomForest(sal_tier ~ ., data = train.df,
                         ntree = 1000,
                         mtry = floor(sqrt(length(selected.var))),
                         importance = TRUE)

pred_rf <- predict(rf_model, newdata = valid.df)
confusionMatrix(pred_rf, valid.df$sal_tier)

#------------------------------------------------------------------------------
# 2 OBB Error 시각화
#------------------------------------------------------------------------------
library(reshape2)
library(ggplot2)

# OOB Error 데이터
err_df <- as.data.frame(rf_model$err.rate)
err_df$Trees <- 1:nrow(err_df)

# long 변환
err_long <- melt(err_df, id.vars = "Trees", variable.name = "Type", value.name = "Error")

# 전체 평균 오류만 필터링
oob_only <- subset(err_long, Type == "OOB")

# 시각화 코드
ggplot(oob_only, aes(x = Trees, y = Error)) +
  geom_line(color = "#E74C3C", linewidth = 1.2) +
  geom_vline(xintercept = oob_only$Trees[which.min(oob_only$Error)], # 최소 OOB Error의 트리 수 
             linetype = "dashed", 
             color = "darkblue", 
             linewidth = 0.5) +
  labs(
    title = "Overall OOB Error by Number of Trees",
    x = "Number of Trees", y = "OOB Error Rate"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    axis.title = element_text(face = "bold")
  )

best_tree <- which.min(rf_model$err.rate[, "OOB"])
best_error <- min(rf_model$err.rate[, "OOB"])

cat(" OOB Error가 가장 낮은 트리 수:", best_tree, "\n")
cat("   ▶ 최소 OOB Error Rate:", round(best_error, 4), "\n")

#------------------------------------------------------------------------------
# 2. 랜덤포레스트 - 교차검증
#------------------------------------------------------------------------------
# 1. 교차검증 설정 (동일하게 유지)
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

# 2. 랜덤포레스트 학습
set.seed(1)
rf_model <- train(sal_tier ~ ., 
                  data = train.df, 
                  method = "rf", 
                  trControl = ctrl,
                  tuneLength = 5,  # mtry 자동 탐색
                  importance = TRUE)

# 3. 결과 확인
print(rf_model)
plot(rf_model)

# 4. 교차검증 정확도 확인
rf_model$resample$Accuracy     # 25개 정확도
summary(rf_model$resample$Accuracy)  # 평균, 중앙값, 범위 등
summary(rf_model$resample$Kappa)     # 평균 Kappa
sd(rf_model$resample$Accuracy)       # 정확도 표준편차

# 5. 박스플롯으로 시각화
boxplot(rf_model$resample$Accuracy, 
        main = "Random Forest Cross-Validation Accuracy", 
        ylab = "Accuracy")

# 교차검증으로 학습한 모델을 검증셋에 적용
pred_rf_cv <- predict(rf_model, newdata = valid.df)

# 혼동 행렬로 평가
confusionMatrix(pred_rf_cv, valid.df$sal_tier)

#------------------------------------------------------------------------------
# 3. ROC-AUC 분석 (One-vs-All)
#------------------------------------------------------------------------------
library(pROC)
library(ggplot2)

# (5-3-1). 테스트셋에 대한 예측 확률 추
rf.prob <- predict(rf_model, newdata = valid.df, type = "prob")
true_labels <- valid.df$sal_tier

# (5-3-2). ROC 곡선 시각화 과정
roc_list <- list()
auc_labels <- c()

for (class in levels(true_labels)) {
  actual_binary <- ifelse(true_labels == class, 1, 0)
  prob <- rf.prob[[class]]
  
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

# ROC 시각화
ggplot(roc_df, aes(x = FPR, y = TPR, color = Class)) +
  geom_line(linewidth = 1.3) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(title = "One-vs-All ROC Curves (Random Forest)",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal() +
  theme(theme(
    legend.title = element_blank(),
    text = element_text(size = 14),
    legend.text = element_text(size = 12)
  ))


#------------------------------------------------------------------------------
# 4. 변수중요도 시각화
#------------------------------------------------------------------------------
library(tidyverse)

varImp(rf_model)  # 변수 중요도 데이터

# (5-4-1). varImp 데이터 가져오기 & long 형 변환
imp_raw <- varImp(rf_model)$importance
imp_raw$Variable <- rownames(imp_raw)

imp_long <- imp_raw %>%
  pivot_longer(cols = -Variable, names_to = "Class", values_to = "Importance")

# 클래스 순서 재정의 (Low → Mid → High)
imp_long$Class <- factor(imp_long$Class, levels = c("Low", "Mid", "High"))

# (5-4-2). 시각화
ggplot(imp_long, aes(x = reorder(Variable, Importance), y = Importance, fill = Class)) +
  geom_col(width = 0.6) +  
  geom_text(aes(label = round(Importance, 1)), hjust = -0.1, size = 3.8) +
  coord_flip() +
  facet_wrap(~ Class) +
  labs(
    title = "클래스별 변수 중요도 (Random Forest)",
    x = "변수", y = "중요도"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    strip.text = element_text(face = "bold", size = 14),
    plot.title = element_text(face = "bold", size = 16),
    axis.title.y = element_text(margin = ggplot2::margin(r = 10)),
    axis.title.x = element_text(margin = ggplot2::margin(t = 10)),
    legend.position = "none"
  ) +
  scale_fill_manual(values = c("Low" = "#2ECC71", "Mid" = "#F1C40F", "High" = "#E74C3C")) +
  ylim(0, max(imp_long$Importance) * 1.15)

#------------------------------------------------------------------------------
# 5. 오분류 심한 선수 분석
#------------------------------------------------------------------------------
# (5-5-1). 이름과 연봉 벡터 추출
valid_names <- nba[valid.index, "Player.Name"]
valid_salary <- nba[valid.index, "Salary"]

# (5-5-2). 오차 계산 및 오차가 심한 상위 5명 추출
misclass_df_salary <- valid.df %>%
  mutate(
    Actual = sal_tier,
    Predicted = pred_rf_cv,
    Player.Name = valid_names,
    True_Salary = valid_salary,
    Pred_Salary = case_when(
      pred_rf_cv == "Low"  ~ 1.2e6,
      pred_rf_cv == "Mid"  ~ 9.3e6,
      pred_rf_cv == "High" ~ 25e6
    ),
    Salary_Error = abs(True_Salary - Pred_Salary)
  ) %>%
  filter(Actual != Predicted) %>%
  arrange(desc(Salary_Error)) %>%
  select(Player.Name, Actual, Predicted, True_Salary, Pred_Salary, Salary_Error) %>%
  head(5)

# 결과 출력
print(misclass_df_salary)


# (5-5-3). 과대예측, 과소예측 선수 1명 
# 과대예측: Darius Garland (Mid -> High)
nba[valid.index, ][valid_names == "Darius Garland",selected.var]

# 과소예측: Al Horford (High -> Mid)
nba[valid.index, ][valid_names == "Al Horford",selected.var]

#------------------------------------------------------------------------------

