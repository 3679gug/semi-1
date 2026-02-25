import pandas as pd
import numpy as np
import re
from statsmodels.formula.api import logit

# =========================
# 1. 데이터 불러오기 및 기본 전처리
# =========================
file_path = r"D:\EDA\vpcs_master_data_v2.csv" 
df = pd.read_csv(file_path, encoding="cp949")

df.columns = [col.strip() for col in df.columns]
outcome = "delivery_pref"
df[outcome] = (df[outcome].astype(str).str.strip().str.lower() == "yes").astype(int)

binary_maps = {
    "Ethic_group": {"킨족": 1, "기타": 0},
    "prev_delivery": {"출산": 1, "미출산": 0},
    "amniotic_fluid": {"정상": 1, "비정상": 0},
}
for col, mp in binary_maps.items():
    if col in df.columns:
        df[col] = df[col].map(mp)

for col in df.columns:
    if df[col].dtype == "object":
        norm = df[col].astype(str).str.strip().str.lower()
        uniq = set(norm.dropna().unique())
        if uniq.issubset({"yes", "no"}):
            df[col] = np.where(norm == "yes", 1, np.where(norm == "no", 0, np.nan))

# =========================
# 2. 연속형 → 범주형 변환
# =========================
def replace_with_cut(df, col, bins, labels):
    if col in df.columns:
        df[col] = pd.cut(df[col], bins=bins, labels=labels)
        df[col] = df[col].astype('category')
        print(f"범주화 완료: {col}")

replace_with_cut(df, "age", [-np.inf, 24, 30, np.inf], ["<25", "25-30", ">30"])
replace_with_cut(df, "BMI", [-np.inf, 24.9, np.inf], ["<25", ">=25"])
replace_with_cut(df, "gestational_age_wk", [-np.inf, 36, 38, np.inf], ["<37", "37-38", ">=39"])
replace_with_cut(df, "fetal_weight_est", [-np.inf, 2499, np.inf], ["<2500", ">=2500"])

categorical_vars = [
    "age", "BMI", "gestational_age_wk", "fetal_weight_est", 
    "occupation", "health_iunsurance", "prev_delivery", "chronic_disease", "anemia", "ivf", "fetal_problem", "amniotic_fluid", "belief_healthy_pregnancy", "belief_vd_ability", "expect_companion"
]

# =========================
# 3. 단변량 분석 실행 (기준 범주 추가 로직 포함)
# =========================
results = []

# 분석에서 제외할 변수 리스트 설정
exclude_vars = ["fear_score", "knowledge_score"]

for col in df.columns:
    # 종속변수이거나 제외 리스트에 포함된 경우 건너뜀
    if col == outcome or col in exclude_vars:
        continue
        
    try:
        data_subset = df[[outcome, col]].dropna()
        if len(data_subset[col].unique()) < 2:
            continue

        # 범주형 여부 판단
        is_categorical = col in categorical_vars or data_subset[col].dtype.name == 'category' or data_subset[col].dtype == 'object'

        # 포뮬러 설정 및 기준 범주 파악
        ref_level = None
        if col == "occupation":
            ref_level = "전업주부"
            formula = f'{outcome} ~ C(occupation, Treatment(reference="{ref_level}"))'
        elif is_categorical:
            # pandas category인 경우 첫 번째 카테고리를 기준으로 사용
            if hasattr(data_subset[col], 'cat'):
                ref_level = data_subset[col].cat.categories[0]
            else:
                ref_level = sorted(data_subset[col].unique())[0]
            formula = f"{outcome} ~ C({col})"
        else:
            formula = f"{outcome} ~ {col}"
        
        # 모델 적합
        res = logit(formula, data=data_subset).fit(method='newton', maxiter=100, disp=False)

        llr_stat = res.llr
        llr_pval = res.llr_pvalue
        pseudo_r2 = res.prsquared

        # --- [추가] 범주형인 경우 기준 범주(Reference)를 먼저 행에 추가 ---
        if is_categorical and ref_level is not None:
            results.append({
                "Variable": col,
                "Level": f"{ref_level} (Ref)",
                "OR": 1.0,
                "95% CI": "-",
                "p_value": "-",
                "LLR": round(llr_stat, 3),
                "LLR_p_value": "<0.001" if llr_pval < 0.001 else round(llr_pval, 3),
                "Pseudo_R2": round(pseudo_r2, 4),
                "N": len(data_subset)
            })

        # 나머지 레벨 결과값 추가
        for term in res.params.index:
            if term == "Intercept":
                continue
            
            coef = res.params[term]
            se = res.bse[term]
            pval = res.pvalues[term]
            OR = np.exp(coef)
            CI_low = np.exp(coef - 1.96 * se)
            CI_high = np.exp(coef + 1.96 * se)
                
            display_name = col
            level_name = ""
            
            # C(variable)[T.level] 형식에서 레벨명 추출
            match = re.search(r"\[T\.(.+)\]", term)
            if match:
                level_name = match.group(1)
            else:
                level_name = "Continuous" # 연속형 변수인 경우

            results.append({
                "Variable": display_name,
                "Level": level_name,
                "OR": round(OR, 3),
                "95% CI": f"{round(CI_low, 3)} - {round(CI_high, 3)}",
                "p_value": "<0.001" if pval < 0.001 else round(pval, 3),
                "LLR": round(llr_stat, 3),
                "LLR_p_value": "<0.001" if llr_pval < 0.001 else round(llr_pval, 3),
                "Pseudo_R2": round(pseudo_r2, 4),
                "N": len(data_subset)
            })
            
    except Exception as e:
        print(f"[{col}] 분석 실패: {e}")

# =========================
# 4. 결과 저장
# =========================
if results:
    result_df = pd.DataFrame(results)
    output_path = r"D:\EDA\Table2_Logistic_Univariate_Results.xlsx"
    result_df.to_excel(output_path, index=False)
    print(f"\n✅ 분석 완료! (fear_score, knowledge_score 제외)")
    print(f"총 {len(result_df)}개의 행이 '{output_path}'에 저장되었습니다.")