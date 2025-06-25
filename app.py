import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("≥CD2術後合併症発生率予測アプリ")

# ✅ モデルファイルを同じディレクトリから読み込む
model_path = "CD2_logistic_model_ensemble.joblib"
cv_models = joblib.load(model_path)  # アンサンブル（5モデルのリスト）

# マッピング定義
sex2_map = {'男性': 1, '女性': 0}
smoking2_map = {'無し': 0, 'あり': 1}
copd2_map = {'無し': 0, 'あり': 1}
location2_map = {'胃': 0, '食道胃接合部': 1}
combined_resection_map = {'無し': 0, 'あり': 1}
asa_ps_2_map = {'1': 0, '2': 1, '3_4': 2}
c_t_3_map = {'cT1': 0, 'cT2': 1, 'cT3': 2, 'cT4': 3}
reconstruction2_map = {'B-1': 0, 'B-2': 1, 'R-Y': 2, 'その他': 3}

# 入力フォーム
st.header("患者情報を入力してください")

age = st.slider("年齢", 20, 100, 60)
sex2 = st.selectbox("性別", list(sex2_map.keys()))
smoking2 = st.selectbox("喫煙歴", list(smoking2_map.keys()))
asa_ps_2 = st.selectbox("ASA-PS", list(asa_ps_2_map.keys()))
copd2 = st.selectbox("COPD", list(copd2_map.keys()))
location2 = st.selectbox("腫瘍の部位", list(location2_map.keys()))
c_t_3 = st.selectbox("cT分類", list(c_t_3_map.keys()))
reconstruction2 = st.selectbox("再建方法", list(reconstruction2_map.keys()))
combined_resection = st.selectbox("合併切除", list(combined_resection_map.keys()))

if st.button("予測する"):
    input_data = pd.DataFrame([{
        'age_per_10': age / 10,
        'sex2': sex2_map[sex2],
        'smoking2': smoking2_map[smoking2],
        'asa_ps_2': asa_ps_2_map[asa_ps_2],
        'copd2': copd2_map[copd2],
        'location2': location2_map[location2],
        'c_t_3': c_t_3_map[c_t_3],
        'reconstruction2': reconstruction2_map[reconstruction2],
        'combined_resection': combined_resection_map[combined_resection]
    }])

    probs = [model.predict_proba(input_data)[0][1] for model in cv_models]
    avg_prob = round(np.mean(probs) * 100, 1)

    st.subheader(f"≥CD2 術後合併症発生確率の予測値：{avg_prob:.1f}%")
