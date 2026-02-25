import pandas as pd
import numpy as np
from scipy import stats

# =========================
# 1. 데이터 로드 및 기본 설정
# =========================
file_path = r"D:\EDA\vpcs_master_data_v2.csv"
df = pd.read_csv(file_path, encoding="cp949")
outcome = "delivery_pref" # 종속변수

# 데이터 전처리: 숫자로 변환 가능한 건 최대한 변환
df = df.apply(pd.to_numeric, errors='ignore')

# =========================
# 2. 변수 분류 설정
# =========================
# 수치형 변수로 처리할 항목 명시
continuous_vars = ["fear_score_std", "knowledge_score_std"]

# 종속변수를 제외한 모든 컬럼 중 수치형이 아닌 것을 범주형으로 간주
all_features = [col for col in df.columns if col != outcome]

# =========================
# 3. 변수 구간화 (Binning)
# =========================
def apply_binning(data, col, bins, labels):
    if col in data.columns:
        data[col] = pd.cut(data[col], bins=bins, labels=labels)
        data[col] = data[col].astype(str).replace('nan', np.nan)
        print(f"✅ 구간화 완료: {col}")

apply_binning(df, "age", [-np.inf, 24, 30, np.inf], ["<25", "25-30", ">30"])
apply_binning(df, "BMI", [-np.inf, 24.9, np.inf], ["<25", ">=25"])
apply_binning(df, "gestational_age_wk", [-np.inf, 36, 38, np.inf], ["<37", "37-38", ">=39"])
apply_binning(df, "fetal_weight_est", [-np.inf, 2499, np.inf], ["<2500", ">=2500"])

# =========================
# 4. 분석 실행 준비
# =========================
# 분석 대상 데이터 필터링 (Outcome이 결측치인 경우 제외)
df = df.dropna(subset=[outcome])

group_yes = df[df[outcome].astype(str).str.upper() == "YES"]
group_no = df[df[outcome].astype(str).str.upper() == "NO"]

# 퍼센트 계산을 위한 그룹별 총 합계
n_yes_total = len(group_yes)
n_no_total = len(group_no)
n_all_total = len(df)

table1_results = []

print("\n--- 모든 변수에 대해 분석 시작 (빈도 및 백분율 포함) ---")

for var in all_features:
    if df[var].dropna().empty:
        continue

    # --- A. 수치형 변수 분석 (t-test) ---
    if var in continuous_vars:
        try:
            # 그룹별 평균 및 표준편차 계산
            mean_y, std_y = group_yes[var].mean(), group_yes[var].std()
            mean_n, std_n = group_no[var].mean(), group_no[var].std()
            mean_total, std_total = df[var].mean(), df[var].std()

            # t-test 수행
            t_stat, p_val = stats.ttest_ind(group_yes[var].dropna(), group_no[var].dropna(), equal_var=False)
            p_text = f"{p_val:.3f}" if p_val >= 0.001 else "<.001"

            table1_results.append({
                "Variable": var,
                "Category": "Mean ± SD",
                "Total (N)": f"{mean_total:.2f} ± {std_total:.2f}",
                "Group_Yes (N)": f"{mean_y:.2f} ± {std_y:.2f}",
                "Group_No (N)": f"{mean_n:.2f} ± {std_n:.2f}",
                "p-value": p_text,
                "Test": "t-test"
            })
        except Exception as e:
            print(f"⚠️ 수치형 변수 '{var}' 분석 중 오류: {e}")
        continue

    # --- B. 범주형 변수 분석 (Chi-square) ---
    try:
        contingency = pd.crosstab(df[var], df[outcome])
        if contingency.size > 0:
            _, p_val, _, _ = stats.chi2_contingency(contingency)
            p_text = f"{p_val:.3f}" if p_val >= 0.001 else "<.001"
        else:
            p_text = "N/A"
    except Exception:
        p_text = "N/A"

    # 고유값 추출
    if df[var].dtype == 'float64':
        unique_vals = sorted(df[var].round(1).dropna().astype(str).unique())
    else:
        unique_vals = sorted(df[var].dropna().astype(str).unique())
    
    first_row = True
    for val in unique_vals:
        # 빈도 계산
        n_y = len(group_yes[group_yes[var].astype(str) == val])
        n_n = len(group_no[group_no[var].astype(str) == val])
        n_total = n_y + n_n
        
        # 백분율 계산 (각 그룹의 전체 수 대비 비율)
        pct_y = (n_y / n_yes_total * 100) if n_yes_total > 0 else 0
        pct_n = (n_n / n_no_total * 100) if n_no_total > 0 else 0
        pct_total = (n_total / n_all_total * 100) if n_all_total > 0 else 0
        
        table1_results.append({
            "Variable": var if first_row else "",
            "Category": val,
            "Total (N)": f"{n_total} ({pct_total:.1f}%)",
            "Group_Yes (N)": f"{n_y} ({pct_y:.1f}%)",
            "Group_No (N)": f"{n_n} ({pct_n:.1f}%)",
            "p-value": p_text if first_row else "",
            "Test": "Chi-square" if first_row else ""
        })
        first_row = False

# =========================
# 5. 엑셀 저장
# =========================
table1_df = pd.DataFrame(table1_results)

# 컬럼 헤더에 각 그룹의 총 N수 명시
final_columns = {
    "Total (N)": f"Total (N={n_all_total})",
    "Group_Yes (N)": f"Group_Yes (N={n_yes_total})",
    "Group_No (N)": f"Group_No (N={n_no_total})"
}
table1_df.rename(columns=final_columns, inplace=True)

save_path = r"D:\EDA\Table1_Descriptive_Statistics.xlsx"

try:
    table1_df.to_excel(save_path, index=False)
    print(f"\n✅ 엑셀 저장 완료: {save_path}")
    print("   (범주형 변수는 N (%) 형식으로 출력되었습니다.)")
except PermissionError:
    print("\n❌ 오류: 엑셀 파일이 열려 있습니다. 닫고 다시 실행하세요.")