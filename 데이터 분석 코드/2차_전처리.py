import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# 1. 데이터 로드
file_path = r'D:\EDA\전치리최종본.csv' 

if os.path.exists(file_path):
    vpcs_v1 = pd.read_csv(file_path, encoding="cp949")
else:
    print(f"파일을 찾을 수 없습니다: {file_path}")
    vpcs_v1 = pd.DataFrame()

if not vpcs_v1.empty:
    # --- 컬럼 리스트 정의 ---
    fear_cols = ['fear_labor_pain', 'fear_episiotomy', 'fear_vd_failure', 'fear_vd_complication']
    knowledge_cols = [
        "vd_short_stay", "vd_less_blood_loss", "vd_better_lochia", "vd_breastfeeding", 
        "vd_less_surgery_risk", "vd_fast_recovery", "vd_skin_to_skin", "vd_future_preg_safe", 
        "vd_lower_cost", "vd_short_interpreg", "vd_less_resp_risk", "vd_early_contact", 
        "vd_microbiota_benefit", "vd_emergency_cs_risk", "vd_instrumental_risk", "vd_postpartum_pain", 
        "cs_avoid_labor_pain", "cs_avoid_long_labor", "cs_reduce_emergency", "cs_avoid_episiotomy", 
        "cs_epidural_risk", "cs_more_blood_loss", "cs_long_stay", "cs_slow_recovery", 
        "cs_prolonged_pain", "cs_breastfeeding_risk", "cs_surgery_risk", "cs_future_risk", 
        "cs_scar_concern", "cs_baby_resp_risk"
    ]
    cs_pc_cols = ["belief_cs_less_pain", "belief_cs_safer_mother", "belief_time_control", "belief_dob_family", "prefer_choose_dob", "cs_avoid_labor_pain", "cs_avoid_long_labor", "cs_reduce_emergency", "cs_avoid_episiotomy", "cs_epidural_risk", "cs_more_blood_loss", "cs_long_stay", "cs_slow_recovery", "cs_prolonged_pain", "cs_breastfeeding_risk", "cs_surgery_risk", "cs_future_risk", "cs_scar_concern", "cs_baby_resp_risk"]
    vd_pc_cols = ["concern_sex_postpartum", "exposed_negative_story", "belief_cs_safer_baby", "vd_short_stay", "vd_less_blood_loss", "vd_better_lochia", "vd_breastfeeding", "vd_less_surgery_risk", "vd_fast_recovery", "vd_skin_to_skin", "vd_future_preg_safe", "vd_lower_cost", "vd_short_interpreg", "vd_less_resp_risk", "vd_early_contact", "vd_microbiota_benefit", "vd_emergency_cs_risk", "vd_instrumental_risk", "vd_postpartum_pain"]
    vd_drawback_cols = ["concern_sex_postpartum", "exposed_negative_story", "vd_emergency_cs_risk", "vd_instrumental_risk", "vd_postpartum_pain"]
    cs_benefit_cols = ["belief_cs_less_pain", "belief_cs_safer_mother", "belief_time_control", "belief_dob_family", "prefer_choose_dob", "family_advice_cs", "provider_advice_cs", "cs_avoid_labor_pain", "cs_avoid_long_labor", "cs_reduce_emergency", "cs_avoid_episiotomy"]

    # [수정] 모든 컬럼 합치기 및 중복 제거
    required_cols = list(set(fear_cols + knowledge_cols + cs_pc_cols + vd_pc_cols + vd_drawback_cols + cs_benefit_cols))

    # [추가] FutureWarning 방지 및 Yes/No 변환
    pd.set_option('future.no_silent_downcasting', True)
    mapping_dict = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}
    
    # 존재하는 컬럼만 처리
    existing_cols = [col for col in required_cols if col in vpcs_v1.columns]
    vpcs_v1[existing_cols] = vpcs_v1[existing_cols].replace(mapping_dict)

    # [중요 수정] 모든 관련 컬럼을 숫자로 변환 (이 부분이 누락되어 TypeError가 발생했음)
    vpcs_v1[existing_cols] = vpcs_v1[existing_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # 3. 점수 계산
    vpcs_v1['fear_score'] = vpcs_v1[fear_cols].sum(axis=1)
    vpcs_v1['knowledge_score'] = vpcs_v1[knowledge_cols].sum(axis=1)
    vpcs_v1['cs_pc_score'] = vpcs_v1[cs_pc_cols].sum(axis=1)
    vpcs_v1['vd_pc_score'] = vpcs_v1[vd_pc_cols].sum(axis=1)
    vpcs_v1['vd_drawback_sum'] = vpcs_v1[vd_drawback_cols].sum(axis=1)
    vpcs_v1['cs_benefit_sum'] = vpcs_v1[cs_benefit_cols].sum(axis=1)

    # 4. 표준화 (Min-Max Scaling)
    scaler = MinMaxScaler()
    vpcs_v1[['fear_score_std', 'knowledge_score_std']] = scaler.fit_transform(vpcs_v1[['fear_score', 'knowledge_score']])

    # 5. 열 삭제
    drop_cols = ["antenatal_class", "yoga_class", "fear_labor_pain", "fear_episiotomy", "fear_vd_failure", "fear_vd_complication", "fear_any", "belief_cs_less_pain", "belief_cs_safer_mother", "concern_sex_postpartum", "belief_time_control", "belief_dob_family", "prefer_choose_dob", "exposed_negative_story", "family_advice_cs", "provider_advice_cs", "belief_cs_safer_baby", "vd_short_stay", "vd_less_blood_loss", "vd_better_lochia", "vd_breastfeeding", "vd_less_surgery_risk", "vd_fast_recovery", "vd_skin_to_skin", "vd_future_preg_safe", "vd_lower_cost", "vd_short_interpreg", "vd_less_resp_risk", "vd_early_contact", "vd_microbiota_benefit", "vd_emergency_cs_risk", "vd_instrumental_risk", "vd_postpartum_pain", "cs_avoid_labor_pain", "cs_avoid_long_labor", "cs_reduce_emergency", "cs_avoid_episiotomy", "cs_epidural_risk", "cs_more_blood_loss", "cs_long_stay", "cs_slow_recovery", "cs_prolonged_pain", "cs_breastfeeding_risk", "cs_surgery_risk", "cs_future_risk", "cs_scar_concern", "cs_baby_resp_risk", "vd_disadvantage_fear", "vd_disadvantage_fear_class", "cs_disadvantage_know", "cs_disadvantage_know_class"]
    vpcs_v1_final = vpcs_v1.drop(columns=drop_cols, errors='ignore')

    # 6. 최종 결과 출력 및 저장
    print("\n" + "="*50)
    print("전처리 완료 및 점수 산출 성공")
    print(f"최종 데이터 형태: {vpcs_v1_final.shape}")
    print("="*50)
    
    save_path = r'D:\EDA\vpcs_master_data_v3.csv'
    vpcs_v1_final.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n파일 저장 완료: {save_path}")