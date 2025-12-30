import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.base import BaseEstimator, TransformerMixin

if "page" not in st.session_state:
        st.session_state.page = "P1"


class NumericFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_feature_names):
        self.numeric_feature_names = numeric_feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.numeric_feature_names)

        X["Temp_Diff"] = (
            X["Process temperature [K]"] - X["Air temperature [K]"]
        )
        X["Power"] = (
            X["Torque [Nm]"] * X["Rotational speed [rpm]"]
        )

        return X.values
    def get_feature_names_out(self, input_features=None):
        return np.array(
            self.numeric_feature_names + ["Temp_Diff", "Power"],
            dtype=object
        )


CSV_PATH = "ai4i2020.csv" 
def page1():
    @st.cache_data
    def load_df(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    @st.cache_data
    def compute_bounds(df: pd.DataFrame, q_low=0.01, q_high=0.99) -> dict:
        num_df = df.select_dtypes(include=["number"])
        qs = num_df.quantile([q_low, 0.5, q_high]).T
        bounds = {}
        for col in qs.index:
            lo = float(qs.loc[col, q_low])
            mid = float(qs.loc[col, 0.5])
            hi = float(qs.loc[col, q_high])
            # 避免 lo==hi 造成 slider 壞掉
            if lo == hi:
                hi = lo + 1.0
            bounds[col] = {"lo": lo, "mid": mid, "hi": hi}
        return bounds

    def slider_manual_missing(label, lo, hi, default, key, step=0.1):
        slider_k = f"{key}_slider"
        manual_k = f"{key}_manual"
        unknown_k = f"{key}_unknown"
        touched_k = f"{key}_touched"
        main_k = key

        # 初始化
        if main_k not in st.session_state:
            st.session_state[main_k] = float(default)
        if slider_k not in st.session_state:
            st.session_state[slider_k] = float(default)
        if manual_k not in st.session_state:
            st.session_state[manual_k] = float(default)
        if touched_k not in st.session_state:
            st.session_state[touched_k] = False

        def _from_slider():
            v = float(st.session_state[slider_k])
            st.session_state[main_k] = v
            st.session_state[manual_k] = v
            st.session_state[touched_k] = True   # ⭐ 主動表態

        def _from_manual():
            v = float(st.session_state[manual_k])
            st.session_state[main_k] = v
            st.session_state[slider_k] = float(np.clip(v, lo, hi))
            st.session_state[touched_k] = True   # ⭐ 主動表態
        
        st.divider()

        # Row 1：slider + manual
        c1, c2 = st.columns([3, 2], vertical_alignment="center")

        with c1:
            st.slider(
                label,
                min_value=float(lo),
                max_value=float(hi),
                step=float(step),
                key=slider_k,
                on_change=_from_slider,
                disabled=st.session_state.get(unknown_k, False),
            )

        with c2:
            st.number_input(
                "Manual",
                step=float(step),
                key=manual_k,
                on_change=_from_manual,
                disabled=st.session_state.get(unknown_k, False),
            )

        # Row 2：unknown
        unknown = st.checkbox("I don't know", key=unknown_k)
        if unknown:
            st.session_state[touched_k] = True   # ⭐ unknown 也是表態
            return np.nan, False, True  # value, out_of_range, touched

        value = float(st.session_state[main_k])
        out_of_range = (value < lo or value > hi)
        touched = st.session_state[touched_k]

        return value, out_of_range, touched


    df = load_df(CSV_PATH)
    bounds = compute_bounds(df, 0.01, 0.99)

    FEATURE_SPECS = [
        {
            "key": "air_temp",
            "label": "Air temperature [K]",
            "col": "Air temperature [K]",
            "step": 0.1,
        },
        {
            "key": "process_temp",
            "label": "Process temperature [K]",
            "col": "Process temperature [K]",
            "step": 0.1,
        },
        {
            "key": "rot_speed",
            "label": "Rotational speed [rpm]",
            "col": "Rotational speed [rpm]",
            "step": 1.0,
        },
        {
            "key": "torque",
            "label": "Torque [Nm]",
            "col": "Torque [Nm]",
            "step": 0.1,
        },
        {
            "key": "tool_wear",
            "label": "Tool wear [min]",
            "col": "Tool wear [min]",
            "step": 1.0,
        },
    ]

    st.title("AI4I 2020 Predictive Maintenance Demo")
    st.write("請輸入機台參數，進行故障風險預測。")
    st.divider()

    type_choice = st.selectbox(
        "Type (L/M/H)",
        options=["— Please select —", "L", "M", "H", "I don't know"],
        index=0,
        key="Type"
    )

    if type_choice == "— Please select —":
        type_val = None
    elif type_choice == "I don't know":
        type_val = np.nan
    else:
        type_val = type_choice

    user_input = {"Type": type_val}

    out_of_range_fields = []

    # col1, col2 = st.columns(2)

    for i,spec in enumerate(FEATURE_SPECS):
        colname = spec["col"]
        lo, mid, hi = bounds[colname]["lo"], bounds[colname]["mid"], bounds[colname]["hi"]

        # target = col1 if i % 2 == 0 else col2
        # with target:
        val, oor , _ = slider_manual_missing(
            label=spec["label"],
            lo=float(lo),
            hi=float(hi),
            default=float(mid),
            key=spec["key"],
            step=float(spec["step"])
            )
        user_input[colname] = val
        if oor:
            out_of_range_fields.append(spec["label"])

    # st.write("Preview:", user_input)


    missing_fields = []

    if type_val is None: 
        missing_fields.append("Type")

    for spec in FEATURE_SPECS:
        key = spec["key"]
        touched = st.session_state.get(f"{key}_touched", False)

        if not touched:
            missing_fields.append(spec["label"])


    #blocked = (len(missing_fields) > 0) or (len(out_of_range_fields) > 0)
    if out_of_range_fields:
        st.error("以下欄位超出建議範圍，建議調整回範圍內：\n\n- " + "\n- ".join(out_of_range_fields))
    
# , disabled=blocked
    if st.button("Veiw my predicition → "):
            if missing_fields:
                st.error("Please fill or mark 'I don't know' for:\n\n- " + "\n- ".join(missing_fields))
                st.stop()

            # 全部通過：存起來帶到 page2
            st.session_state["p1_data"] = user_input
            for k in list(st.session_state.keys()):
                if k in ["overview_artifacts"] or k.startswith("artifacts_"):
                    del st.session_state[k]
            st.session_state["page"] = "P2"
            st.rerun()

# ====== P2 helpers ======
FAILURE_SHORT = {
    "Tool wear failure": "Tool wear",
    "Heat dissipation failure": "Heat dissipation",
    "Power failure": "Power",
    "Overstrain failure": "Overstrain",
    "Random failure": "Random",
}
   

@st.cache_data
def load_background(n=400):
    df = pd.read_csv(CSV_PATH)

    # AI4I 常見非特徵欄位（有就丟）
    drop_candidates = ["UDI", "Product ID", "Target", "Failure Type"]
    drop_cols = [c for c in drop_candidates if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if len(df) > n:
        df = df.sample(n, random_state=42)
    return df

@st.cache_resource
def load_pipeline(pkl_path: str):
    return joblib.load(pkl_path)

def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def compute_one_failure_artifacts(pipe, user_df, bg_df):
    """
    回傳：
      proba: float
      exp_user: shap.Explanation (單筆 waterfall 用)
      exp_bg: shap.Explanation (beeswarm 用)
    """
    proba = float(pipe.predict_proba(user_df)[0, 1])

    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]  # XGBClassifier

    X_user = pre.transform(user_df)
    X_bg = pre.transform(bg_df)

    explainer = shap.TreeExplainer(model)
    sv_user = explainer(X_user)  # Explanation（新版 shap）
    sv_bg = explainer(X_bg)

    feature_names = pre.get_feature_names_out() if hasattr(pre, "get_feature_names_out") else None

    X_user_dense = _to_dense(X_user)[0]
    X_bg_dense = _to_dense(X_bg)

    exp_user = shap.Explanation(
        values=sv_user.values[0],
        base_values=sv_user.base_values[0],
        data=X_user_dense,
        feature_names=feature_names
    )

    exp_bg = shap.Explanation(
        values=sv_bg.values,
        base_values=sv_bg.base_values,
        data=X_bg_dense,
        feature_names=feature_names
    )

    return proba, exp_user, exp_bg

def risk_style(proba: float):
    """
    回傳： (label, color, bg)
    你可以自己調 threshold
    """
    if proba < 0.33:
        return "LOW RISK / 低風險", "#16a34a", "#ecfdf5"
    elif proba < 0.66:
        return "MEDIUM RISK / 中度風險", "#f59e0b", "#fffbeb"
    else:
        return "HIGH RISK / 高風險", "#dc2626", "#fef2f2"

def render_risk_card(title: str, proba: float):
    label, color, bg = risk_style(proba)
    st.markdown(
        f"""
        <div style="
            border:2px solid {color};
            border-radius:16px;
            padding:18px 16px;
            background:{bg};
            min-height:170px;
            display:flex;
            flex-direction:column;
            justify-content:space-between;
        ">
            <div style="font-weight:700; font-size:16px; color:#111827;">
                {title}
            </div>
            <div style="font-weight:800; font-size:34px; color:{color};">
                {proba*100:.1f}%
            </div>
            <div style="font-weight:700; letter-spacing:1px; color:{color};">
                {label}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



FAILURE_OVERVIEW_PKL = "ai4i_xgb_pipeline_final.pkl"

def page2():
    # --- Back to P1 ---
    if st.button("← Back / 回到輸入頁"):
        st.session_state["page"] = "P1"
        st.rerun()

    st.markdown("<h1 style='text-align:center;'>Machine Failure Overview / 總故障機率</h1>", unsafe_allow_html=True)
    st.divider()

    data = st.session_state.get("p1_data")
    if data is None:
        st.warning("No input data found. Please go back to Page 1.")
        st.stop()

    user_df = pd.DataFrame([data])
    bg_df = load_background(n=400)

    # --- Load overall pipeline ---
    pipe = load_pipeline(FAILURE_OVERVIEW_PKL)
    

    # --- Prediction + SHAP artifacts (cache) ---
    cache_key = "overview_artifacts"
    
    if cache_key not in st.session_state:
        proba, exp_user, exp_bg = compute_one_failure_artifacts(pipe, user_df, bg_df)
        st.session_state[cache_key] = {"proba": proba, "exp_user": exp_user, "exp_bg": exp_bg}
    proba = float(st.session_state[cache_key]["proba"])
    

    # --- Risk card (1 card only) ---
    st.subheader("Failure Probability / 故障機率")
    # st.write("DEBUG — predict():", pipe.predict(user_df))
    # st.write("DEBUG — predict_proba():", pipe.predict_proba(user_df))
    # st.write("DEBUG — proba used:", proba)
    render_risk_card("Machine failure (overall)", proba)

    # --- Gate to next step ---
    # 你可以自己調門檻：例如 0.5 / 0.4 / 0.3
    FAIL_GATE = 0.7
    likely_failure = (proba >= FAIL_GATE)

    if not likely_failure:
        st.info(f"模型粗估：目前較不可能發生故障（p={proba:.2%} < {FAIL_GATE:.0%}）。")
        # st.info(f"模型粗估：目前較不可能發生故障（p={proba:.2%} < {FAIL_GATE:.0%}）。若仍想看五種 failure，可自行降低門檻或提供『強制繼續』選項。")

    st.divider()

    # --- Waterfall / Force toggle (Force coming soon) ---
    st.markdown("## 🔎 Why this result? / 原因分析")

    tab_waterfall, tab_force = st.tabs([
        "Waterfall Plot (Factor Contribution) / 風險累積圖",
        "Force Plot (Risk Push/Pull) / 風險拔河圖"
    ])

    with tab_waterfall:
        exp_user = st.session_state[cache_key]["exp_user"]
        fig = plt.figure()
        shap.plots.waterfall(exp_user, max_display=15, show=False)
        st.pyplot(fig, clear_figure=True)

    with tab_force:
        exp_user = st.session_state[cache_key]["exp_user"]

        # shap.force_plot 需要 feature_names（沒有也可以，但會變成 feature_0,1,...）
        fn = exp_user.feature_names if exp_user.feature_names is not None else None

        force = shap.force_plot(
            exp_user.base_values,     # 期望值（base value）
            exp_user.values,          # 各特徵 SHAP 值
            exp_user.data,            # 該筆資料的特徵值
            feature_names=fn,
            matplotlib=False
        )

        # ✅ Streamlit 用 HTML 方式嵌入（最像你截圖那種互動 force）
        components.html(
            f"""
            <div>{shap.getjs()}</div>
            <div>{force.html()}</div>
            """,
            height=320,
            scrolling=True
        )


    st.divider()

    # --- Beeswarm (global) ---
    st.markdown("## 📊 Global Explanation / 模型整體解釋")
    exp_bg = st.session_state[cache_key]["exp_bg"]
    tab_bee, tab_bar = st.tabs([
        "Beeswarm / 特徵影響力",
        "Bar / 重要性排名"
    ])

    with tab_bee:
        fig = plt.figure()
        shap.plots.beeswarm(exp_bg, max_display=15, show=False)
        st.pyplot(fig, clear_figure=True)

    with tab_bar:
        fig = plt.figure()
        # bar 就是 summary_plot 的 bar 版本（或用 shap.plots.bar）
        shap.summary_plot(
            exp_bg.values,
            features=exp_bg.data,
            feature_names=exp_bg.feature_names,
            plot_type="bar",
            max_display=15,
            show=False
        )
        st.pyplot(fig, clear_figure=True)


    st.divider()

if st.session_state.page == "P1":
    page1()
elif st.session_state.page == "P2":
    page2()

