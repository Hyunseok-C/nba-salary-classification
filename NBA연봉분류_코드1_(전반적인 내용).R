#------------------------------------------------------------------------------
## 데이터마이닝 프로젝트 최종 코딩 - 최현석, 황성진
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 코드1 - 데이터 전처리 및 시각화와 변수선택, 최종 모델 시각화
#------------------------------------------------------------------------------
# 0. 데이터불러오기
#------------------------------------------------------------------------------
nba <- read.csv("C:\\Users\\chs02\\OneDrive\\바탕 화면\\데마프로젝트\\NBA_Player_Salaries_(2022-23).csv")

#------------------------------------------------------------------------------
# 1. 전처리
#------------------------------------------------------------------------------
# (1-1-1) ID 변수 제거
nba <- nba[, -1]

# (1-1-2) 결측치
# 결측치 시각화
na_info <- sapply(nba, function(x) c(sum = sum(is.na(x)), rate = mean(is.na(x))))
na_df <- as.data.frame(t(na_info))
na_df <- na_df[na_df$sum > 0, ]  # 결측치가 존재하는 변수만 선택
na_df$variable <- rownames(na_df)

library(ggplot2)
ggplot(na_df, aes(x = reorder(variable, -rate), y = sum)) +
  geom_bar(stat = "identity", fill = "lightcoral") +
  geom_text(aes(label = paste0(sum, " (", round(rate * 100, 1), "%)")),
            vjust = -0.5, size = 3.2) +
  labs(title = "Missing Value Count Rate by Variable",
       x = "Variable",
       y = "Missing Count") +
  theme_minimal()

# 결측치 제거
nba <- na.omit(nba)

# (1-1-3) 혼합 포지션을 단일 포지션으로 
nba$Position <- sub("-.*", "", nba$Position)

#------------------------------------------------------------------------------
# 2. 변수 추가: 연봉 분류
#------------------------------------------------------------------------------
salary_cap = 123655000
nba$pct_cap <- (as.numeric(nba$Salary) / salary_cap) * 100

bins <- c(0, 2, 13, 100)
labels <- c("Low", "Mid", "High")
nba$sal_tier <- cut(nba$pct_cap, breaks = bins, labels = labels, right = FALSE)

table(nba$sal_tier)

#------------------------------------------------------------------------------
# 3. 데이터 시각화
#------------------------------------------------------------------------------
## (1-3-1) 시각화1: 구간별 히스토그램
library(ggplot2)
ggplot(nba, aes(x = pct_cap, fill = sal_tier)) +
  geom_histogram(bins = 20, color = "white", alpha = 0.9) +
  labs(title = "Distribution of Salary Cap Percentage",
       x = "Salary as % of Cap",
       y = "Count",
       fill = "Salary Tier") +
  theme_minimal() +
  scale_fill_manual(values = c("Low" = "lightgreen",
                               "Mid" = "orange",
                               "High" = "darkred"))

## (1-3-2) 시각화2: 변수별 상관계수 히트맵
library(ggplot2)
library(reshape2)

# 1) 수치형 변수 상관계수 행렬 계산
numeric_vars <- nba[sapply(nba, is.numeric)] 
numeric_vars <- numeric_vars[, !(names(numeric_vars) %in% "pct_cap")]

cor_mat <- cor(numeric_vars, use = "complete.obs")

# 2) 연봉과의 상관계수 기준 변수 정렬
salary_corr <- cor_mat[, "Salary"]
ordered_vars <- names(sort(abs(salary_corr), decreasing = TRUE))
cor_mat <- cor_mat[ordered_vars, ordered_vars]

# 3) 하삼각 제거 및 변수 순서 고정
cor_mat[lower.tri(cor_mat)] <- 0
cor_melt <- melt(cor_mat, na.rm = TRUE) # long-format으로
cor_melt$Var1 <- factor(cor_melt$Var1, levels = ordered_vars)
cor_melt$Var2 <- factor(cor_melt$Var2, levels = ordered_vars)

# 4) 상삼각 히트맵 시각화
ggplot(cor_melt, aes(x = Var2, y = Var1, fill = value)) +
  geom_tile(color = rgb(0, 0, 0, 0.2), linewidth = 0.3) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0, limits = c(-1, 1), name = "Correlation") +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  coord_fixed() +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.4, size = 5),
    axis.text.y = element_text(size = 5),
    plot.margin = ggplot2::margin(t = 0, r = 0, b = 0, l = 0)
  ) +
  labs(
    title = "Correlation Heatmap (Variables Ordered by Correlation with Salary)",
    x = NULL,
    y = NULL
  )

## (1-3-3) 시각화3: 연봉 구간별 주요변수 박스플롯
# 1. 선수 지표 - 연차
ggplot(nba, aes(x = sal_tier, y = Years.of.Service, fill = sal_tier)) +
  geom_boxplot() +
  scale_fill_manual(name = "연봉분류",
                    values = c("Low" = "lightgreen",
                               "Mid" = "khaki1",
                               "High" = "lightcoral")) +
  labs(title = "연봉 구간별 연차",
       x = "연봉 구간",
       y = "연차") +
  theme_minimal()

# 2. 공격 지표 - 득점수
ggplot(nba, aes(x = sal_tier, y = PPG, fill = sal_tier)) +
  geom_boxplot() +
  scale_fill_manual(name = "연봉분류",
                    values = c("Low" = "lightgreen",
                               "Mid" = "khaki1",
                               "High" = "lightcoral")) +
  labs(title = "연봉 구간별 평균 득점수",
       x = "연봉 구간",
       y = "평균 득점수") +
  theme_minimal()

# 3. 수비 지표
ggplot(nba, aes(x = sal_tier, y = DRB, fill = sal_tier)) +
  geom_boxplot() +
  scale_fill_manual(name = "연봉분류",
                    values = c("Low" = "lightgreen",
                               "Mid" = "khaki1",
                               "High" = "lightcoral")) +
  labs(title = "연봉 구간별 수비 리바운드 ",
       x = "연봉 구간",
       y = "수비 리바운드") +
  theme_minimal()

#------------------------------------------------------------------------------
# 4. 변수선택 - Lasso & Elastic Net
#------------------------------------------------------------------------------
# (1-4-0) 도메인 지식과 시각화 등을 통해 비율 지표와 같은 변수 제외
# (1-4-1) 데이터 분할
selected.var <- c("Years.of.Service","Position","Age","GP","GS"
                  ,"MP","FG","FGA",	"X3P",	"X3PA",	"X2P","X2PA","FT","FTA",
                  "ORB","DRB","TRB","AST","STL","BLK","TOV","PF","PPG","Total.Minutes","PER",
                  "TS.","USG.","OWS","DWS","WS","OBPM","DBPM","BPM","VORP")
set.seed(1)
idx <- sample(nrow(nba), nrow(nba)*0.8)

train.df <- nba[idx, c("Salary", selected.var)]
valid.df <- nba[-idx, c("Salary", selected.var)]

x_train <- model.matrix(Salary ~ ., data = train.df)[, -1]
y_train <- train.df$Salary
x_valid <- model.matrix(Salary ~ ., data = valid.df)[, -1]
y_valid <- valid.df$Salary

#------------------------------------------------------------------------------
# (1-4-2) Elastic Net의 최적 alpha 찾기 (lambda.1se 기준)
library(glmnet)
library(Metrics)

alpha_list <- seq(0.1, 0.9, by = 0.1)
rmse_list <- numeric(length(alpha_list))

for (i in seq_along(alpha_list)) {
  a <- alpha_list[i]
  set.seed(1)
  cv_model <- cv.glmnet(x_train, y_train, alpha = a)
  lambda_1se <- cv_model$lambda.1se
  pred <- predict(cv_model, newx = x_valid, s = lambda_1se)
  rmse_list[i] <- rmse(y_valid, pred)
}

alpha_rmse_df <- data.frame(alpha = alpha_list, RMSE = rmse_list)
best_alpha <- alpha_rmse_df$alpha[which.min(alpha_rmse_df$RMSE)]

# 최적 알파 값 = 0.9
cat("lastic Net 최적 alpha:", best_alpha, "\n")
print(alpha_rmse_df)

#------------------------------------------------------------------------------
# (1-4-3) Lasso & Elastic Net 성능 비교 (lambda.min 기준 비교는 유지)
library(ggplot2)
library(dplyr)
library(tidyr)

lambda_seq <- 10^seq(5.5, -5.5, length.out = 100)

lasso_rmses <- sapply(lambda_seq, function(lam) {
  model <- glmnet(x_train, y_train, alpha = 1, lambda = lam)
  pred <- predict(model, newx = x_valid, s = lam)
  rmse(y_valid, pred)
})

elnet_rmses <- sapply(lambda_seq, function(lam) {
  model <- glmnet(x_train, y_train, alpha = best_alpha, lambda = lam)
  pred <- predict(model, newx = x_valid, s = lam)
  rmse(y_valid, pred)
})

# 데이터 정리
rmse_long <- data.frame(
  log_lambda = log10(lambda_seq),
  Lasso = lasso_rmses,
  ElasticNet = elnet_rmses
) %>%
  pivot_longer(cols = c("Lasso", "ElasticNet"), 
                names_to = "Model", 
                values_to = "RMSE")

# 라쏘와 엘라스틱 넷 성능 비교 시각화 (확실한 비교를 위해 x축 조정)
ggplot(rmse_long, aes(x = log_lambda, y = RMSE, color = Model)) +
  geom_line(size = 1.2) +
  labs(
    title = paste("RMSE by lambda: Lasso vs Elastic Net (α =", best_alpha, ")"),
    x = "log(lambda)",
    y = "RMSE"
  ) +
  scale_x_continuous(limits = c(3, 5.5)) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.title = element_blank()
  )
# 라쏘가 엘라스틱보다 성능이 더 좋음 -> 라쏘 선택

#------------------------------------------------------------------------------
# (1-4-4) 변수 선택 결과 출력 (lambda.1se 기준 적용)
## 1. 라쏘 
# 1) 라쏘 최종 CV 모델
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)
lambda_1se_lasso <- cv_lasso$lambda.1se
pred_lasso <- predict(cv_lasso, newx = x_valid, s = lambda_1se_lasso)
rmse_lasso_1se <- rmse(y_valid, pred_lasso)

# 2) 라쏘 변수 선택
coef_lasso <- coef(cv_lasso, s = lambda_1se_lasso)
selected_vars_lasso <- rownames(coef_lasso)[coef_lasso[, 1] != 0][-1]

## 2. 엘라스틱
# 1) Elastic Net 최종 CV 모델
cv_elnet <- cv.glmnet(x_train, y_train, alpha = best_alpha)
lambda_1se_elnet <- cv_elnet$lambda.1se
pred_elnet <- predict(cv_elnet, newx = x_valid, s = lambda_1se_elnet)
rmse_elnet_1se <- rmse(y_valid, pred_elnet)

# 2) Elastic Net 변수 선택
coef_elnet <- coef(cv_elnet, s = lambda_1se_elnet)
selected_vars_elnet <- rownames(coef_elnet)[coef_elnet[, 1] != 0][-1]

## 3. 라쏘와 엘라스틱 최종 결과
results <- list(
  Lasso = list(
    RMSE = rmse_lasso_1se,
    Lambda = lambda_1se_lasso,
    Selected_Variables_Count = length(selected_vars_lasso),
    Selected_Variables = selected_vars_lasso
  ),
  Elastic_Net = list(
    RMSE = rmse_elnet_1se,
    Alpha = best_alpha,
    Lambda_1se = lambda_1se_elnet,
    Selected_Variables_Count = length(selected_vars_elnet),
    Selected_Variables = selected_vars_elnet
  )
)
print(results)

## 4. 최종 선택된 변수 (9개)
results$Lasso$Selected_Variables

#------------------------------------------------------------------------------
# 5. 모델 별 성능
# KNN, 나이브베이즈, 결정트리, 랜덤포레스트는 코드를 따로 작성
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# 6. 모델 최종 비교 시각화
#------------------------------------------------------------------------------
library(ggplot2)
library(dplyr)
library(tidyr)

#------------------------------------------------------------------------------
# (1.6.1). 모델별 오분류율 시각화 (교차검증, 검증셋 기준)
#------------------------------------------------------------------------------

# 데이터 생성
df <- data.frame(
  Model = c("KNN", "Naive Bayes", "Decision Tree", "Random Forest"),
  CV_Accuracy = c(0.7340, 0.6439, 0.6549, 0.7260),
  Validation_Accuracy = c(0.7614, 0.6512, 0.7500, 0.7386)
)

# 오분류율 계산
df <- df %>%
  mutate(CV_Error = 1 - CV_Accuracy,
         Validation_Error = 1 - Validation_Accuracy)

# 모델 순서 지정
df$Model <- factor(df$Model, levels = c("KNN", "Random Forest", "Decision Tree", "Naive Bayes"))

# long format
df_long <- df %>%
  pivot_longer(cols = c(CV_Error, Validation_Error),
               names_to = "Type", values_to = "Error")

# 그래프
ggplot(df_long, aes(x = Model, y = Error, group = Model)) +
  geom_line(color = "gray60") +
  geom_point(aes(color = Type), size = 4) +
  geom_text(aes(label = round(Error, 3), color = Type), 
            vjust = -1, size = 4.2, show.legend = FALSE) +
  scale_color_manual(values = c("CV_Error" = "#e41a1c", "Validation_Error" = "#377eb8"),
                     labels = c("CV Error", "Validation Error")) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.15))) +
  labs(title = "CV Error vs Validation Error by Model",
       x = "Model",
       y = "Error Rate",
       color = NULL) +
  theme_minimal(base_size = 14) +
  theme(
    panel.border = element_rect(color = "black", fill = NA),
    plot.title = element_text(hjust = 0.5)
  )


#------------------------------------------------------------------------------
# (1.6.2). 모델별 클래스 오분류율 시각화
#------------------------------------------------------------------------------
# 데이터프레임 생성
df <- data.frame(
  Model = rep(c("KNN", "Random Forest", "Decision Tree", "Naive Bayes"), each = 3),
  Class = rep(c("Low", "Mid", "High"), times = 4),
  Misclassification = c(
    0.1600, 0.3023, 0.2000,       # KNN
    0.2800, 0.2791, 0.2000,       # Random Forest
    0.2000, 0.3023, 0.2000,       # Decision Tree
    0.3871, 0.3590, 0.2500        # Naive Bayes
  )
)

df$Class <- factor(df$Class, levels = c("High", "Mid", "Low"))

ggplot(df, aes(x = Class, y = Misclassification, group = Model, color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_text(aes(label = round(Misclassification, 3)), 
            vjust = -0.8, size = 3.5, show.legend = FALSE) +  # 여기에 추가
  labs(title = "모델별 클래스 오분류율",
       x = "클래스", y = "오분류율 (1 - 민감도)", color = "모델") +
  theme_minimal(base_size = 13)

#------------------------------------------------------------------------------
