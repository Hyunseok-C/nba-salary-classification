#------------------------------------------------------------------------------
## 데이터마이닝 프로젝트 최종 코딩 - 최현석, 황성진
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 코드2 - KNN 모델
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
# 1. KNN 모델
# - 기존 KNN 모델: 한번 실행 (교차검증X)
# kknn(kernel = "rectangular", distance = 2) ≈ class::knn()
#------------------------------------------------------------------------------
# (2-1-1). 가변수 변환
feature_cols <- setdiff(names(train.df), "sal_tier") # 종속변수 제외
x_train <- model.matrix(~ . -1, data = train.df[, feature_cols])
x_valid <- model.matrix(~ . -1, data = valid.df[, feature_cols])

# (2-1-2). 정규화
train.scale <- scale(x_train)
valid.scale <- scale(x_valid,
                     center = attr(train.scale, "scaled:center"),
                     scale = attr(train.scale, "scaled:scale"))


# (2-1-3). 최적의 k 선택
k_values <- seq(1, 20, by = 2)
accuracy.df <- data.frame(k = k_values, accuracy = rep(0, length(k_values)))

set.seed(1)
for(i in seq_along(k_values)) {
  k <- k_values[i]
  set.seed(1)
  knn.pred <- class::knn(train = train.scale,
                         test = valid.scale,
                         cl = train.df$sal_tier,
                         k = k)
  accuracy.df[i, 2] <- mean(knn.pred == valid.df$sal_tier)
}

best_k <- accuracy.df$k[which.max(accuracy.df$accuracy)] # 최적의 홀수 k
best_k

library(ggplot2)
ggplot(accuracy.df, aes(x = k, y = accuracy)) +
  geom_line() +
  geom_point() +
  geom_vline(xintercept = best_k, linetype = "dashed", color = "red") +
  ggtitle("KNN Accuracy vs. K")

# (2-1-4). 예측 KNN 수행
knn.pred <- class::knn(train = train.scale,
                       test = valid.scale,
                       cl = train.df$sal_tier,
                       k = best_k)

# (2-1-5). 분류 성능 지표
cm <- caret::confusionMatrix(knn.pred, valid.df$sal_tier)
cm

#-----------------------------------------------------------------------------
# 2. KNN 모델 
# - 가중치 모델 비교 (6가지), 최적의 커널 선택
#-----------------------------------------------------------------------------
library(caret)
library(kknn)
set.seed(1)

# (2-2-1). 공통 교차검증 설정
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

# (2-2-2). 커널 및 거리 조합 정의
kernels <- c("rectangular", "triangular", "epanechnikov", 
             "gaussian", "inv", "optimal")

# (2-2-3). 각 커널과 거리별 모델 학습
results <- list()
for (k in kernels) {
  cat("Running kernel:", k, "\n")
  
  # 튜닝 그리드 설정
  grid <- expand.grid(
    kmax = seq(1, 13, by = 2),
    distance = 2,
    kernel = k
  )
  
  # 모델 학습
  model <- train(sal_tier ~ ., 
                 data = train.df,
                 method = "kknn",
                 trControl = ctrl,
                 tuneGrid = grid)
  
  # 예측
  pred <- predict(model, newdata = valid.df)
  cm <- confusionMatrix(pred, valid.df$sal_tier)
  
  # 결과 저장
  results[[k]] <- list(model = model,
                       accuracy_cv = summary(model$resample$Accuracy),
                       sd_cv = sd(model$resample$Accuracy),
                       cm = cm)
}

# 결과 출력
for (k in kernels) {
  cat("- Kernel:", k, "\n")
  print(results[[k]]$accuracy_cv) # 25개 교차검증 결과 요약
  cat("SD:", results[[k]]$sd_cv, "\n") # 교차검증 정확도 표준편차
  cat("Test Accuracy:", results[[k]]$cm$overall["Accuracy"], "\n\n") # 테스트셋 성능
}

# (2-2-4) 결과 시각화
# 데이터 프레임
acc_df <- data.frame(
  Kernel = kernels,
  CV_Mean = sapply(kernels, function(k) mean(results[[k]]$model$resample$Accuracy)),
  CV_SD   = sapply(kernels, function(k) sd(results[[k]]$model$resample$Accuracy)),
  Test_Acc = sapply(kernels, function(k) results[[k]]$cm$overall["Accuracy"])
); acc_df

# CV 평균 정확도 기준 정렬
acc_df$Kernel <- factor(acc_df$Kernel, levels = acc_df$Kernel[order(-acc_df$CV_Mean)])

# 시각화
ggplot(acc_df, aes(x = Kernel, y = CV_Mean)) +
  geom_bar(stat = "identity", fill = "lightblue", color = "black", width = 0.6) +
  geom_errorbar(aes(ymin = CV_Mean - CV_SD, ymax = CV_Mean + CV_SD), 
                width = 0.2, color = "blue") +
  geom_point(aes(y = Test_Acc), color = "red", size = 3) +
  labs(title = "커널별 교차검증 정확도 (±SD) 및 테스트 정확도",
       y = "정확도",
       x = "커널") +
  theme_minimal() +
  coord_cartesian(ylim = c(0.65, 0.78))

#-----------------------------------------------------------------------------
# 3. KNN 모델 
# - 각 거리별 비교 (2에서 최적인 커널로 선정된 gussian으로 고정)
#-----------------------------------------------------------------------------
library(caret)
library(kknn)
set.seed(1)

# (2-3-1) 교차검증 설정
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

# (2-3-2) 거리 차수 정의 
distances <- c(1, 2, 3) # 1: Manhattan, 2: Euclidean, 3: Minkowski

# (2-3-3) 거리별 모델 학습 (kernel은 gaussian으로 고정)
results <- list()
for (d in distances) {
  cat("Running distance:", d, "\n")
  
  grid <- expand.grid(
    kmax = seq(1, 13, by = 2),
    distance = d,
    kernel = "gaussian"
  )
  
  model <- train(sal_tier ~ ., 
                 data = train.df,
                 method = "kknn",
                 trControl = ctrl,
                 tuneGrid = grid)
  
  pred <- predict(model, newdata = valid.df)
  cm <- confusionMatrix(pred, valid.df$sal_tier)
  
  results[[as.character(d)]] <- list(
    model = model,
    accuracy_cv = summary(model$resample$Accuracy),
    sd_cv = sd(model$resample$Accuracy),
    cm = cm
  )
}

# (2-3-4) 결과 시각화
# 데이터프레임
acc_df <- data.frame(
  Distance = as.factor(distances),
  CV_Mean  = sapply(results, function(x) mean(x$model$resample$Accuracy)),
  CV_SD    = sapply(results, function(x) sd(x$model$resample$Accuracy)),
  Test_Acc = sapply(results, function(x) x$cm$overall["Accuracy"])
); acc_df

# 정렬: CV 정확도 기준
acc_df$Distance <- factor(acc_df$Distance, levels = acc_df$Distance[order(-acc_df$CV_Mean)])

# 시각화
ggplot(acc_df, aes(x = Distance, y = CV_Mean)) +
  geom_bar(stat = "identity", fill = "lightblue", color = "black", width = 0.6) +
  geom_errorbar(aes(ymin = CV_Mean - CV_SD, ymax = CV_Mean + CV_SD),
                width = 0.2, color = "blue") +
  geom_point(aes(y = Test_Acc), color = "red", size = 3) +
  labs(title = "거리 차수별 교차검증 정확도 (±SD) 및 테스트 정확도",
       y = "정확도",
       x = "거리 차수 (Minkowski p)") +
  theme_minimal() +
  coord_cartesian(ylim = c(0.65, 0.78))

#-----------------------------------------------------------------------------
# 3. KNN 최종 모델 
# - 교차검증 5-fold 5회 반복 (자동으로 최적의 k, kernel=gaussian, distance = 1)
#-----------------------------------------------------------------------------
# (2-3-1). 공통 교차검증 설정
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

# (2-3-2). 튜닝 그리드
grid <- expand.grid(
  kmax = seq(1, 25, by = 2),
  distance = 1,
  kernel = "gaussian" # 최적의 가중치 모델 수정
)

# (2-3-3). 모델 학습 및 출력
set.seed(1)
knn_model <- train(
  sal_tier ~ .,
  data = train.df,
  method = "kknn",
  trControl = ctrl,
  tuneGrid = grid
)
print(knn_model)

# (2-3-4). 정확도 요약 통계
summary(knn_model$resample$Accuracy)
sd(knn_model$resample$Accuracy)
boxplot(knn_model$resample$Accuracy, main = "KNN Cross-Validation Accuracy", ylab = "Accuracy")

# (2-3-5). 검증셋 예측 및 평가
pred <- predict(knn_model, newdata = valid.df)
caret::confusionMatrix(pred, valid.df$sal_tier)

#------------------------------------------------------------------------------
# 4. ROC-AUC 분석 (One-vs-All)
#------------------------------------------------------------------------------
library(pROC)
library(ggplot2)

# (2-4-1). 예측 확률과 실제 라벨
prob_knn <- predict(knn_model, newdata = valid.df, type = "prob")
true_labels <- valid.df$sal_tier

# (2-4-2). ROC 곡선 시각화 과정
roc_list <- list()
auc_labels <- c()

for (class in levels(true_labels)) {
  actual_binary <- ifelse(true_labels == class, 1, 0)
  prob <- as.numeric(prob_knn[, class])
  
  roc_obj <- roc(actual_binary, prob, quiet = TRUE)
  roc_list[[class]] <- roc_obj
  
  auc_val <- auc(roc_obj)
  auc_labels <- c(auc_labels, paste0(class, " (AUC = ", round(auc_val, 3), ")"))
}

# 3. 데이터프레임 변환
roc_df <- do.call(rbind, lapply(seq_along(roc_list), function(i) {
  roc_curve <- roc_list[[i]]
  data.frame(
    FPR = 1 - rev(roc_curve$specificities),
    TPR = rev(roc_curve$sensitivities),
    Class = auc_labels[i]
  )
}))

# 4. 시각화
ggplot(roc_df, aes(x = FPR, y = TPR, color = Class)) +
  geom_line(linewidth = 1.3) +  
  geom_abline(linetype = "dashed", color = "gray") +
  labs(title = "One-vs-All ROC Curves (KNN)",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal() +
  theme(theme(
    legend.title = element_blank(),
    text = element_text(size = 14),       
    legend.text = element_text(size = 12) 
  ))

#------------------------------------------------------------------------------

