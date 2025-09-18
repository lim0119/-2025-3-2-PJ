# -2025-3-2-PJ
# 📌 뇌 MRI 기반 알츠하이머 보조진단 모델 성능 평가(1)


# 0. 라이브러리 설치
!pip install scikit-learn matplotlib seaborn pandas numpy

# 1. 라이브러리 불러오기
# 성능지표 계산(sklearn), 데이터처리(pandas, numpy), 시각화(matplotlib, seaborn)
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, confusion_matrix, f1_score,
    balanced_accuracy_score, matthews_corrcoef, brier_score_loss, roc_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 2. CSV 파일 업로드
from google.colab import files
uploaded = files.upload()  # 업로드할 CSV 선택(CSV=데이터셋)

# 업로드한 파일명 가져오기
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)
print("Columns in dataset:", df.columns.tolist())

# 3. 컬럼 설정 (필수/선택)
# 반드시 필요한 컬럼: 실제 라벨(label), 모델 예측 확률(pred_prob)
# 추가적으로 환자 ID, 시점(timepoint), subgroup 있으면 분석 확장 가능
y_true = df['label'].values            # 실제 라벨
y_pred_prob = df['pred_prob'].values   # 모델 예측 확률 (0~1)

patient_id = df['patient_id'].values if 'patient_id' in df.columns else None
timepoint = df['timepoint'].values if 'timepoint' in df.columns else None
subgroup = df['subgroup'].values if 'subgroup' in df.columns else None

# ================================
# 4. 성능 지표 계산(AUC / PR-AUC)
#  모델 성능을 수치로 확인 (AUC, PR-AUC, Sensitivity, Specificity 등)
# - AUC: 모델이 환자/비환자 구분을 얼마나 잘하는지
# - PR-AUC: 불균형 데이터(환자 수 적음)에 적합한 지표
# - Sensitivity(민감도): 환자를 잘 잡아내는 비율
# - Specificity(특이도): 정상인을 잘 구분하는 비율
# - F1: Precision과 Recall의 조화 평균
auc_score = roc_auc_score(y_true, y_pred_prob)
precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
pr_auc_score = auc(recall, precision)
print(f"AUC: {auc_score:.3f}, PR-AUC: {pr_auc_score:.3f}")

# Threshold 기반 지표
threshold = 0.5  # 임계값 (0.5 이상이면 알츠하이머로 분류)
y_pred_label = (y_pred_prob >= threshold).astype(int)
cm = confusion_matrix(y_true, y_pred_label)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0
specificity = tn / (tn + fp) if (tn+fp) > 0 else 0
f1 = f1_score(y_true, y_pred_label)
print(f"Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}, F1: {f1:.3f}")

# 추가 지표
# - Balanced Accuracy: 민감도+특이도 평균
# - MCC: 양성/음성 균형 고려한 상관계수
# - Brier Score: 예측확률과 실제 정답의 차이
bal_acc = balanced_accuracy_score(y_true, y_pred_label)
mcc = matthews_corrcoef(y_true, y_pred_label)
brier = brier_score_loss(y_true, y_pred_prob)
print(f"Balanced Acc: {bal_acc:.3f}, MCC: {mcc:.3f}, Brier Score: {brier:.3f}")

# ================================
# 5. 부트스트랩 95% CI 계단
# 샘플을 여러 번 재추출해서(bootstrap) 지표의 신뢰구간 계산
# 단순 점수뿐만 아니라 "불확실성" 범위도 확인 가능
def bootstrap_ci(y_true, y_pred_prob, n_bootstrap=1000, alpha=0.05):
    rng = np.random.default_rng(seed=42)
    metrics_dict = {"AUC": [], "PR-AUC": [], "Sensitivity": [], 
                    "Specificity": [], "F1": [], "Balanced Acc": []}

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        y_true_bs = y_true[idx]
        y_prob_bs = y_pred_prob[idx]
        y_label_bs = (y_prob_bs >= 0.5).astype(int)

        if len(np.unique(y_true_bs)) < 2:
            continue

        try:
            auc_bs = roc_auc_score(y_true_bs, y_prob_bs)
            precision_bs, recall_bs, _ = precision_recall_curve(y_true_bs, y_prob_bs)
            pr_auc_bs = auc(recall_bs, precision_bs)
        except:
            continue

        cm_bs = confusion_matrix(y_true_bs, y_label_bs)
        if cm_bs.shape == (2,2):
            tn, fp, fn, tp = cm_bs.ravel()
            sens_bs = tp / (tp+fn) if (tp+fn) > 0 else 0
            spec_bs = tn / (tn+fp) if (tn+fp) > 0 else 0
        else:
            sens_bs, spec_bs = np.nan, np.nan

        f1_bs = f1_score(y_true_bs, y_label_bs)
        bal_bs = balanced_accuracy_score(y_true_bs, y_label_bs)

        metrics_dict["AUC"].append(auc_bs)
        metrics_dict["PR-AUC"].append(pr_auc_bs)
        metrics_dict["Sensitivity"].append(sens_bs)
        metrics_dict["Specificity"].append(spec_bs)
        metrics_dict["F1"].append(f1_bs)
        metrics_dict["Balanced Acc"].append(bal_bs)

    # CI 계산
    ci_results = {}
    for metric, values in metrics_dict.items():
        if len(values) > 0:
            lower = np.percentile(values, 100*alpha/2)
            upper = np.percentile(values, 100*(1-alpha/2))
            ci_results[metric] = (np.mean(values), lower, upper)
    return ci_results

ci_results = bootstrap_ci(y_true, y_pred_prob)
print("\n📌 Bootstrap 95% CI 결과")
for metric, (mean, lower, upper) in ci_results.items():
    print(f"{metric}: {mean:.3f} (95% CI {lower:.3f} ~ {upper:.3f})")

# ================================
# 6. 시각화
# ROC, PR, Confusion Matrix, Calibration Curve 시각화
# - ROC: 모델이 환자/비환자 구분하는 능력
# - PR Curve: 소수 클래스(환자)에 대한 모델 성능 강조
# - Confusion Matrix: 실제/예측 비교 (오분류 확인)
# - Calibration Curve: 예측 확률과 실제 발생률 일치 여부 확인
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC={auc_score:.3f})")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# PR Curve
plt.figure()
plt.plot(recall, precision, label=f"PR-AUC={pr_auc_score:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# Confusion Matrix heatmap
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["NC","AD"], yticklabels=["NC","AD"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Calibration Curve
prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")
plt.title("Calibration Curve")
plt.show()

# ================================
# 7. Calibration Error (ECE)
# 예측 확률이 실제와 얼마나 차이가 나는지 수치화
# (낮을수록 모델의 확률값 신뢰 가능)
bin_counts, _ = np.histogram(y_pred_prob, bins=10)
ece = np.sum(bin_counts / len(y_pred_prob) * np.abs(prob_true - prob_pred))
print(f"ECE: {ece:.3f}")

# ================================
# 8. Uncertainty (예시)
# 불확실성 값 예시 (실제로는 Bayesian, MC Dropout 등 필요)
# - 모델이 자신 없는 샘플(high uncertainty) 탐지 가능
uncertainty = np.random.rand(len(y_pred_prob)) * 0.1
high_uncertainty_idx = np.where(uncertainty > 0.08)[0]
print(f"High uncertainty samples: {high_uncertainty_idx}")

# ================================
# 9. Subgroup 성능
# 특정 그룹(성별, 나이대 등)별 성능 차이 분석
# - 공정성/편향성 확인
if subgroup is not None:
    df_sub = pd.DataFrame({'y_true':y_true, 'y_pred_prob':y_pred_prob, 'subgroup':subgroup})
    for g in df_sub['subgroup'].unique():
        g_df = df_sub[df_sub['subgroup']==g]
        if len(g_df['y_true'].unique()) > 1:
            g_auc = roc_auc_score(g_df['y_true'], g_df['y_pred_prob'])
            print(f"Subgroup {g} AUC: {g_auc:.3f}")

# ================================
# 10. Longitudinal Consistency
# 한 환자를 시간(timepoint) 따라 추적 → 예측값의 일관성 체크
# - 갑자기 큰 변화가 있으면 모델 안정성 문제 가능
if patient_id is not None and timepoint is not None:
    df_long = pd.DataFrame({'patient_id':patient_id, 'timepoint':timepoint, 'y_pred_prob':y_pred_prob})
    for pid in df_long['patient_id'].unique():
        series = df_long[df_long['patient_id']==pid].sort_values('timepoint')['y_pred_prob'].values
        if len(series) > 1:
            change = np.diff(series)
            if np.any(np.abs(change) > 0.5):
                print(f"Patient {pid} has abrupt prediction change: {series}")

# ================================
# 11. Quality Check (QC)
# 데이터 품질 검증
# - 결측치, 0~1 범위 밖 확률값 여부 확인
df_qc = pd.DataFrame({'y_pred_prob':y_pred_prob})
missing_count = df_qc.isna().sum().values[0]
out_of_bounds = np.sum((df_qc['y_pred_prob']<0) | (df_qc['y_pred_prob']>1))
print(f"Missing values: {missing_count}, Out-of-bounds predictions: {out_of_bounds}")
