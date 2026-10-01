import streamlit as st
import joblib
import pandas as pd
import numpy as np

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Height & Weight Predictor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM UI STYLING
# =========================================================

st.markdown("""<style>

/* ---------- APP BACKGROUND ---------- */

.stApp {
    background:
        radial-gradient(circle at 15% 5%, rgba(37, 99, 235, 0.22), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(124, 58, 237, 0.18), transparent 25%),
        linear-gradient(145deg, #07111f 0%, #0b1220 45%, #101827 100%);
    color: white;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}


/* ---------- HIDE DEFAULT STREAMLIT ELEMENTS ---------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* ---------- HEADINGS ---------- */

h1 {
    color: #f8fafc !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -1.5px !important;
}

h2, h3 {
    color: #f8fafc !important;
}

p {
    color: #cbd5e1;
}


/* ---------- BORDER CONTAINERS ---------- */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 22px !important;
    box-shadow: 0px 18px 55px rgba(0, 0, 0, 0.20);
    backdrop-filter: blur(14px);
}


/* ---------- METRIC CARDS ---------- */

[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.045);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px;
    padding: 18px 20px;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-weight: 750;
}


/* ---------- BUTTONS ---------- */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    min-height: 46px;

    border: 1px solid rgba(96, 165, 250, 0.25);

    background:
        linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.22),
            rgba(124, 58, 237, 0.18)
        );

    color: #e2e8f0;
    font-weight: 650;

    transition: all 0.20s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    border-color: #60a5fa;
    color: white;
    box-shadow: 0px 8px 25px rgba(37, 99, 235, 0.25);
}


/* ---------- SLIDER ---------- */

.stSlider label {
    color: #e2e8f0 !important;
    font-weight: 650 !important;
}


/* ---------- NUMBER INPUT ---------- */

.stNumberInput label {
    color: #e2e8f0 !important;
    font-weight: 650 !important;
}

.stNumberInput input {
    background: rgba(15, 23, 42, 0.95) !important;
    color: white !important;
    border-radius: 10px !important;
}


/* ---------- TABS ---------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 10px;
    color: #94a3b8;
    padding-left: 18px;
    padding-right: 18px;
}

.stTabs [aria-selected="true"] {
    background: rgba(37, 99, 235, 0.18) !important;
    color: white !important;
}


/* ---------- PROGRESS ---------- */

.stProgress > div > div > div > div {
    background: linear-gradient(
        90deg,
        #2563eb,
        #8b5cf6
    );
}


/* ---------- DATAFRAME ---------- */

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}


/* ---------- CAPTION ---------- */

[data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}


/* ---------- DIVIDER ---------- */

hr {
    border-color: rgba(148, 163, 184, 0.12);
}

</style>""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("best_model_height_weight.pkl")


model = load_model()


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_weight(height):

    if hasattr(model, "feature_names_in_"):

        feature_name = model.feature_names_in_[0]

        data = pd.DataFrame({
            feature_name: [height]
        })

    else:

        data = np.array([[height]])

    prediction = model.predict(data)

    return float(
        np.asarray(prediction).reshape(-1)[0]
    )


# =========================================================
# RANDOM FOREST TREE SPREAD
# =========================================================

def get_tree_spread(height):

    if not hasattr(model, "estimators_"):
        return None

    x = np.array([[height]])

    predictions = []

    for tree in model.estimators_:
        predictions.append(
            float(tree.predict(x)[0])
        )

    return float(np.std(predictions))


# =========================================================
# SESSION STATE
# =========================================================

if "height" not in st.session_state:
    st.session_state.height = 170.0

if "history" not in st.session_state:
    st.session_state.history = []


def set_height(value):
    st.session_state.height = float(value)


# =========================================================
# HERO
# =========================================================

st.title("⚖️ Height → Weight Predictor")

st.write(
    "An interactive machine-learning dashboard that estimates weight "
    "from height using a trained Random Forest regression model."
)

st.caption(
    "Adjust the height below and the prediction updates instantly."
)


# =========================================================
# TOP INFORMATION STRIP
# =========================================================

st.write("")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "🤖 Model",
        "Random Forest"
    )

with c2:
    st.metric(
        "📏 Input",
        "Height"
    )

with c3:
    st.metric(
        "⚖️ Output",
        "Weight"
    )

with c4:
    st.metric(
        "⚡ Mode",
        "Live"
    )


st.write("")


# =========================================================
# MAIN DASHBOARD
# =========================================================

left, right = st.columns(
    [0.9, 1.1],
    gap="large"
)


# =========================================================
# LEFT SIDE - CONTROLS
# =========================================================

with left:

    with st.container(border=True):

        st.subheader("🎛️ Prediction Controls")

        st.caption(
            "Choose a preset or fine-tune the height manually."
        )

        st.write("")

        st.write("##### Quick Heights")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.button(
                "150 cm",
                on_click=set_height,
                args=(150,)
            )

        with p2:
            st.button(
                "160 cm",
                on_click=set_height,
                args=(160,)
            )

        with p3:
            st.button(
                "170 cm",
                on_click=set_height,
                args=(170,)
            )

        p4, p5, p6 = st.columns(3)

        with p4:
            st.button(
                "180 cm",
                on_click=set_height,
                args=(180,)
            )

        with p5:
            st.button(
                "190 cm",
                on_click=set_height,
                args=(190,)
            )

        with p6:
            st.button(
                "200 cm",
                on_click=set_height,
                args=(200,)
            )

        st.divider()

        height = st.slider(
            "Height",
            min_value=120.0,
            max_value=220.0,
            step=0.5,
            key="height",
            format="%.1f cm"
        )

        st.caption(
            f"Selected height: {height:.1f} cm"
        )

        feet_total = height / 30.48

        feet = int(feet_total)

        inches = (
            feet_total - feet
        ) * 12

        st.metric(
            "Height Conversion",
            f"{feet} ft {inches:.1f} in"
        )


# =========================================================
# PREDICT
# =========================================================

prediction = predict_weight(height)

prediction = max(
    0.0,
    prediction
)

tree_spread = get_tree_spread(height)


# =========================================================
# RIGHT SIDE - RESULT
# =========================================================

with right:

    with st.container(border=True):

        st.subheader("✨ Prediction Result")

        st.caption(
            "The model output changes live with the selected height."
        )

        st.write("")

        st.metric(
            label="🎯 Predicted Weight",
            value=f"{prediction:.2f} kg",
            delta=f"Height: {height:.1f} cm",
            delta_color="off"
        )

        st.write("")

        progress = min(
            max(prediction / 120, 0.0),
            1.0
        )

        st.progress(progress)

        st.caption(
            "Visual scale based on a 0–120 kg display range."
        )

        st.divider()

        r1, r2 = st.columns(2)

        with r1:

            st.metric(
                "📐 Selected Height",
                f"{height:.1f} cm"
            )

        with r2:

            if tree_spread is not None:

                st.metric(
                    "🌲 Tree Spread",
                    f"± {tree_spread:.2f} kg"
                )

            else:

                st.metric(
                    "Model Status",
                    "Ready"
                )

        st.info(
            "💡 This is a machine-learning estimate based only on "
            "the height patterns learned from the training dataset."
        )

        if st.button(
            "➕ Add this prediction to comparison",
            use_container_width=True
        ):

            item = {
                "Height (cm)": round(height, 1),
                "Predicted Weight (kg)": round(prediction, 2)
            }

            if item not in st.session_state.history:

                st.session_state.history.append(item)

                st.toast(
                    "Prediction added to comparison."
                )

            else:

                st.toast(
                    "This prediction is already saved."
                )


# =========================================================
# ANALYTICS AREA
# =========================================================

st.write("")

st.subheader("📊 Explore the Model")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Prediction Curve",
        "🔍 Compare Heights",
        "📋 Saved Predictions",
        "🧠 Model Details"
    ]
)


# =========================================================
# TAB 1 - PREDICTION CURVE
# =========================================================

with tab1:

    st.write(
        "See how the model's predicted weight changes "
        "across different heights."
    )

    curve_heights = np.arange(
        120,
        221,
        2
    )

    curve_weights = [
        predict_weight(h)
        for h in curve_heights
    ]

    curve_df = pd.DataFrame({
        "Height (cm)": curve_heights,
        "Predicted Weight (kg)": curve_weights
    })

    curve_df = curve_df.set_index(
        "Height (cm)"
    )

    st.line_chart(
        curve_df,
        height=360
    )

    st.caption(
        "The graph shows the learned relationship between "
        "height and predicted weight."
    )


# =========================================================
# TAB 2 - HEIGHT COMPARISON
# =========================================================

with tab2:

    st.write(
        "Select multiple heights and compare their predictions."
    )

    selected_heights = st.multiselect(
        "Choose heights",
        options=list(
            range(130, 211, 5)
        ),
        default=[
            150,
            160,
            170,
            180,
            190
        ]
    )

    if selected_heights:

        comparison = pd.DataFrame({
            "Height (cm)": selected_heights,
            "Predicted Weight (kg)": [
                round(
                    predict_weight(h),
                    2
                )
                for h in selected_heights
            ]
        })

        chart_data = comparison.set_index(
            "Height (cm)"
        )

        st.bar_chart(
            chart_data,
            height=340
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Select at least one height to compare."
        )


# =========================================================
# TAB 3 - SAVED PREDICTIONS
# =========================================================

with tab3:

    if len(
        st.session_state.history
    ) == 0:

        st.info(
            "No saved predictions yet. "
            "Use 'Add this prediction to comparison' above."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        history_chart = (
            history_df
            .set_index("Height (cm)")
        )

        st.line_chart(
            history_chart,
            height=300
        )

        if st.button(
            "🗑️ Clear Saved Predictions"
        ):

            st.session_state.history = []

            st.rerun()


# =========================================================
# TAB 4 - MODEL DETAILS
# =========================================================

with tab4:

    st.write(
        "Technical information about the prediction system."
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Model Type",
            type(model).__name__
        )

    with m2:

        if hasattr(
            model,
            "n_estimators"
        ):

            st.metric(
                "Decision Trees",
                model.n_estimators
            )

        else:

            st.metric(
                "Estimator",
                "Regression"
            )

    with m3:

        if hasattr(
            model,
            "n_features_in_"
        ):

            st.metric(
                "Input Features",
                model.n_features_in_
            )

        else:

            st.metric(
                "Input Features",
                "1"
            )

    st.divider()

    st.write("#### Prediction Pipeline")

    st.code(
        "Height (cm)  →  Random Forest Model  →  Predicted Weight (kg)",
        language=None
    )

    st.success(
        "✅ This saved model expects raw height values. "
        "No scaler is used during prediction."
    )

    st.write("#### Technologies")

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        st.metric(
            "Language",
            "Python"
        )

    with tech2:
        st.metric(
            "Interface",
            "Streamlit"
        )

    with tech3:
        st.metric(
            "ML",
            "Scikit-learn"
        )

    with tech4:
        st.metric(
            "Data",
            "Pandas"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "⚖️ Height → Weight Predictor  •  "
    "Machine Learning Regression Demo  •  "
    "Python + Streamlit + Scikit-learn"
)

st.caption(
    "Predictions are generated from the training dataset and "
    "are intended for demonstration, not medical assessment."
)