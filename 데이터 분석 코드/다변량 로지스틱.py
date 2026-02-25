import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
# =========================
# 1. 데이터 불러오기
# =========================
schools = pd.read_csv(
    r"D:\EDA\vpcs_master_data_v2.csv",
    encoding="cp949"
)
outcome = "delivery_pref"
schools[outcome] = (schools[outcome] == "Yes").astype(int)
# =========================
# 2. 이진 변수 매핑
# =========================
binary_maps = {
    "Ethic_group": {"킨족": 1, "기타": 0},
    "prev_delivery": {"출산": 1, "미출산": 0},
    "amniotic_fluid": {"정상": 1, "비정상": 0},
}
for col, mp in binary_maps.items():
    if col in schools.columns:
        schools[col] = schools[col].map(mp)
# =========================
# 3. Yes / No → 0 / 1 (occupation 보호)
# =========================
for col in schools.columns:
    if schools[col].dtype == "object":
        norm = schools[col].astype(str).str.strip().str.lower()
        uniq = set(norm.dropna().unique())
        if uniq.issubset({"yes", "no"}):
            schools[col] = np.where(norm == "yes", 1,
                                     np.where(norm == "no", 0, np.nan))
# =========================
# 4. 연속형 → 범주형
# =========================
def replace_with_cut(df, col, bins, labels):
    pos = df.columns.get_loc(col)
    new = pd.cut(df[col], bins=bins, labels=labels)
    df.drop(columns=[col], inplace=True)
    df.insert(pos, col, new)
replace_with_cut(schools, "age",
                 [-np.inf, 24, 30, np.inf],
                 ["<25", "25-30", ">30"])
replace_with_cut(schools, "BMI",
                 [-np.inf, 24.9, np.inf],
                 ["<25", ">=25"])
replace_with_cut(schools, "gestational_age_wk",
                 [-np.inf, 36, 38, np.inf],
                 ["<37", "37-38", ">=39"])
replace_with_cut(schools, "fetal_weight_est",
                 [-np.inf, 2499, np.inf],
                 ["<2500", ">=2500"])
categorical_vars = [
    "age", "BMI", "gestational_age_wk", "fetal_weight_est", 
    "occupation", "health_iunsurance", "prev_delivery", "chronic_disease", "anemia", "ivf", "fetal_problem", "amniotic_fluid", "belief_healthy_pregnancy", "belief_vd_ability", "expect_companion"
]

# =========================
# 4-1. 특정 변수 반올림 (추가)
# =========================
if 'knowledge_score_std' in schools.columns:
    schools['knowledge_score_std'] = schools['knowledge_score_std'].round(3)

# =========================
# 5. 블록 함수
# =========================
def make_block(df, start, end):
    s = df.columns.get_loc(start)
    e = df.columns.get_loc(end)
    return df.columns[s:e + 1].tolist()
block1 = make_block(schools, "age", "health_insurance")
block2 = make_block(schools, "BMI", "amniotic_fluid")
block3 = ["belief_healthy_pregnancy", "belief_vd_ability", "fear_score_std", "expect_companion", "knowledge_score_std"]

# =========================
# 6. 수정된 변수 제거 함수 (예외 처리 추가)
# =========================
dropped_vars_log = []

def drop_separation_vars(df, y, vars_, model_name, exempt_vars=None):
    if exempt_vars is None:
        exempt_vars = []
        
    safe = []
    for v in vars_:
        if v == y: continue
        
        # 💡 예외 리스트에 있는 변수는 검사 없이 통과
        if v in exempt_vars:
            safe.append(v)
            continue
            
        try:
            ct = pd.crosstab(df[v], df[y])
            if (ct == 0).any().any():
                dropped_vars_log.append({
                    "Model": model_name,
                    "Variable": v,
                    "Reason": "Perfect separation (zero cell)"
                })
            else:
                safe.append(v)
        except Exception as e:
            # 에러 발생 시(연속형 변수 등) 일단 포함시키도록 안전장치
            safe.append(v)
    return safe
# 💡 Model 3를 구성할 때 knowledge_score_std를 예외로 설정
block2_safe = drop_separation_vars(schools, outcome, block2, "Model 2")
#block3_safe = drop_separation_vars(
    #schools, outcome, block3, "Model 3", 
    #exempt_vars=["knowledge_score_std"] # 👈 이 변수는 절대 빼지 마라!)
models = {
    "Model 1": block1,
    "Model 2": block1 + block2_safe,
    "Model 3": block1 + block2_safe + block3
}
# =========================
# 7. Table 3 로지스틱 회귀
# =========================
table3_results = []
model_fit_stats = []
for model_name, vars_in_model in models.items():
    try:
        rhs = []
        for v in vars_in_model:
            if v == outcome:
                continue
            if v == "occupation":
                rhs.append(
                    'C(occupation, Treatment(reference="전업주부"))'
                )
            elif v in categorical_vars:
                rhs.append(f"C({v})")
            else:
                rhs.append(v)
        formula = f"{outcome} ~ " + " + ".join(rhs)
        #model = logit(formula, data=schools).fit(disp=False)
        # 💡 fit(method='bfgs') 옵션을 추가하여 복잡한 모델의 수렴을 돕습니다.
        model = logit(formula, data=schools).fit(method='bfgs', maxiter=100, disp=False)
        model_fit_stats.append({
            "Model": model_name,
            "N": int(model.nobs),
            "Pseudo_R2": round(model.prsquared, 4),
            "LLR": round(model.llr, 3),
            "LLR_p_value": (
                "<0.001" if model.llr_pvalue < 0.001
                else round(model.llr_pvalue, 4)
            )
        })
        for term in model.params.index:
            if term == "Intercept":
                continue
            coef, se, p = (
                model.params[term],
                model.bse[term],
                model.pvalues[term]
            )
            table3_results.append({
                "Model": model_name,
                "Variable": term,
                "OR": f"{np.exp(coef):.3f}",
                "95% CI": f"{np.exp(coef-1.96*se):.3f} – {np.exp(coef+1.96*se):.3f}",
                "p_value": "<0.001" if p < 0.001 else f"{p:.3f}"
            })
        print(f":두꺼운_확인_표시: {model_name} 완료")
    except Exception as e:
        print(f":x: {model_name} 실패 → {e}")
# =========================
# 8. Excel 저장 (통합본)
# =========================
file_name = "Table3_Logistic_Multivariate_Results.xlsx"

with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
    pd.DataFrame(table3_results).to_excel(writer, sheet_name="Logistic_Models", index=False)
    pd.DataFrame(model_fit_stats).to_excel(writer, sheet_name="Model_Fit", index=False)
    pd.DataFrame(dropped_vars_log).to_excel(writer, sheet_name="Dropped_Variables", index=False)

print(f"\n✅ 엑셀 파일 생성 완료: {file_name}")
print(" - 시트명: Logistic_Models, Model_Fit, Dropped_Variables")