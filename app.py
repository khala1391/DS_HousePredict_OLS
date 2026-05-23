import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pydeck as pdk
import streamlit as st
from train import preprocess

MODEL_PATH   = "model/model.pkl"
COLUMNS_PATH = "model/columns.json"
EVAL_PATH    = "model/eval_data.pkl"
DATA_PATH    = "data/bengaluru_house_prices.csv"
COORDS_PATH  = "data/location_coords.csv"

SQFT_MIN = 300
SQFT_MAX = 10000
MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

ANALYSIS_PAGES = [
    ("map",         "🗺️ แผนที่ทำเล"),
    ("scatter",     "📐 พื้นที่ vs ราคา"),
    ("boxplot",     "📦 ราคาตามทำเล"),
    ("actual_pred", "🎯 Actual vs Predicted"),
    ("residuals",   "📉 Residuals"),
]


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(COLUMNS_PATH) as f:
        columns = json.load(f)["data_columns"]
    return model, columns


@st.cache_data
def load_eval():
    return joblib.load(EVAL_PATH)


@st.cache_data
def load_processed_data():
    return preprocess(pd.read_csv(DATA_PATH))


@st.cache_data
def load_geodata():
    return pd.read_csv(COORDS_PATH)


def predict_price(model, columns, location, sqft, bath, bhk):
    data = {col: 0 for col in columns}
    data["total_sqft"] = sqft
    data["bath"]       = bath
    data["bhk"]        = bhk
    if location in data:
        data[location] = 1
    return round(model.predict(pd.DataFrame([data]))[0], 2)


st.set_page_config(page_title="Bangalore House Price Prediction", page_icon="🏠", layout="wide")
st.markdown("""
<style>
  .block-container { padding-top: 4rem !important; padding-bottom: 0.5rem !important; }
  div[data-testid="stVerticalBlock"] > div { gap: 0.3rem; }
  div[data-testid="stCaptionContainer"] { margin-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

_title_col, _link_col = st.columns([4, 1.2])
with _title_col:
    st.markdown("### 🏠 Bangalore House Price Prediction")
    st.caption("Linear Regression · Bangalore real estate data · R² = 0.78")
with _link_col:
    st.markdown(
        '<div style="display:flex;justify-content:flex-end;padding-top:0.3rem">'
        '<a href="https://www.linkedin.com/in/yuttapong-m/" target="_blank" '
        'style="display:inline-flex;align-items:center;gap:6px;background:#0A66C2;'
        'color:#FFFFFF;padding:6px 14px;border-radius:6px;text-decoration:none;'
        'font-size:0.85rem;font-weight:600;white-space:nowrap;">'
        '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="white" viewBox="0 0 24 24">'
        '<path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 '
        '2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 '
        '5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 '
        '13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 '
        '24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>'
        '</svg>'
        'yuttapong-m</a></div>',
        unsafe_allow_html=True,
    )

# ── Session state ───────────────────────────
if "pred_location"  not in st.session_state:
    st.session_state.pred_location  = None   # set after geo_df loads
if "analysis_page"  not in st.session_state:
    st.session_state.analysis_page  = "map"

model, columns = load_model()
geo_df_base    = load_geodata()
geo_locations  = sorted(geo_df_base["location"].tolist())

if st.session_state.pred_location is None:
    st.session_state.pred_location = geo_locations[0]

tab1, tab2 = st.tabs(["🔮 ทำนายราคา", "📊 วิเคราะห์ข้อมูล"])

# ═══════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1.6])

    with right:
        geo_df = geo_df_base.copy()
        geo_df["selected"] = geo_df["location"] == st.session_state.pred_location
        geo_df["color"]    = geo_df["selected"].apply(
            lambda s: [255, 90, 0, 240] if s else [60, 120, 210, 180]
        )
        geo_df["radius"]   = geo_df["selected"].apply(lambda s: 650 if s else 350)
        geo_df["label"]    = geo_df.apply(
            lambda r: f"📍 {r['location']}" if r["selected"] else r["location"], axis=1
        )

        event = st.pydeck_chart(
            pdk.Deck(
                layers=[pdk.Layer(
                    "ScatterplotLayer", id="loc-layer", data=geo_df,
                    get_position="[lon, lat]", get_fill_color="color",
                    get_radius="radius", pickable=True, auto_highlight=True,
                )],
                initial_view_state=pdk.ViewState(
                    latitude=12.97, longitude=77.59, zoom=10.2, pitch=0
                ),
                map_style=MAP_STYLE,
                tooltip={"text": "{label}"},
            ),
            on_select="rerun", selection_mode="single-object",
            use_container_width=True, height=290,
        )

        if event and event.selection:
            clicked = event.selection.get("objects", {}).get("loc-layer", [])
            if clicked:
                st.session_state.pred_location = clicked[0]["location"]

        st.caption("คลิกจุดบนแผนที่เพื่อเลือกทำเล")

    with left:
        st.markdown(f"**📍 ทำเลที่เลือก:** {st.session_state.pred_location}")

        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)

        with r1c1:
            sqft = st.number_input(
                "Total Square Feet",
                min_value=float(SQFT_MIN), max_value=float(SQFT_MAX),
                value=1000.0, step=50.0,
                help=f"พื้นที่รวม ({SQFT_MIN}–{SQFT_MAX} sqft)"
            )
        with r1c2:
            bhk = st.number_input(
                "BHK (Bedrooms)", min_value=1, max_value=10, value=2, step=1,
                help="Bedroom Hall Kitchen"
            )
        with r2c1:
            bath = st.number_input(
                "Bathrooms", min_value=1, max_value=10, value=2, step=1,
                help="ปกติไม่ควรเกิน BHK + 2"
            )
        with r2c2:
            st.markdown('<div style="height:1.6rem"></div>', unsafe_allow_html=True)
            predict = st.button("Predict Price", type="primary", use_container_width=True)

        if predict:
            warns = []
            if bath > bhk + 2:
                warns.append("ห้องน้ำมากกว่า BHK+2 — อาจกระทบความแม่นยำ")
            if sqft / bhk < 300:
                warns.append("พื้นที่ต่อห้องน้อยกว่า 300 sqft")
            for w in warns:
                st.warning(w)
            price = predict_price(model, columns, st.session_state.pred_location, sqft, bath, bhk)
            st.success(f"ราคาประมาณ: **{price} Lakh INR**")
            st.caption(f"≈ ₹{price * 100_000:,.0f}")


# ═══════════════════════════════════════════
# TAB 2 — ANALYSIS  (3-column layout)
# ═══════════════════════════════════════════
with tab2:
    df        = load_processed_data()
    eval_data = load_eval()
    y_test    = eval_data["y_test"]
    y_pred    = eval_data["y_pred"]
    residuals = y_test - y_pred
    mean_err  = residuals.mean()
    mae       = np.abs(residuals).mean()

    nav_col, chart_col, info_col = st.columns([1, 2.8, 1.5])

    # ── Column 1: vertical nav buttons ─────
    with nav_col:
        st.markdown("**เลือกการวิเคราะห์**")
        for page_id, page_name in ANALYSIS_PAGES:
            is_active = st.session_state.analysis_page == page_id
            if st.button(
                page_name,
                key=f"nav_{page_id}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state.analysis_page = page_id
                st.rerun()

    page = st.session_state.analysis_page

    # ── Column 2: chart / map ───────────────
    with chart_col:

        if page == "map":
            geo = geo_df_base.copy()
            p_min, p_max = geo["median_price"].min(), geo["median_price"].max()
            geo["color"] = geo["median_price"].apply(
                lambda p: [
                    int(255 * (p - p_min) / (p_max - p_min + 1e-9)),
                    int(100 * (1 - (p - p_min) / (p_max - p_min + 1e-9))),
                    int(200 * (1 - (p - p_min) / (p_max - p_min + 1e-9))),
                    180,
                ]
            )
            geo["radius"] = (geo["median_price"] / p_max) * 1800 + 300
            geo["tooltip_text"] = geo.apply(
                lambda r: f"{r['location']}\nMedian: {r['median_price']:.0f} Lakh\nTx: {int(r['count'])}",
                axis=1,
            )
            st.pydeck_chart(
                pdk.Deck(
                    layers=[pdk.Layer(
                        "ScatterplotLayer", data=geo,
                        get_position="[lon, lat]", get_fill_color="color",
                        get_radius="radius", pickable=True, auto_highlight=True,
                    )],
                    initial_view_state=pdk.ViewState(
                        latitude=12.97, longitude=77.59, zoom=10.5, pitch=0
                    ),
                    map_style=MAP_STYLE,
                    tooltip={"text": "{tooltip_text}"},
                ),
                height=320,
                use_container_width=True,
            )

        elif page == "scatter":
            df_plot = df[df["total_sqft"] < df["total_sqft"].quantile(0.99)]
            df_plot = df_plot[df_plot["price"] < df_plot["price"].quantile(0.99)]
            df_plot = df_plot[df_plot["bhk"].isin([1, 2, 3, 4, 5])]
            palette = {1: "#4C72B0", 2: "#55A868", 3: "#C44E52", 4: "#8172B2", 5: "#CCB974"}
            fig, ax = plt.subplots(figsize=(5.5, 3))
            for bhk_val, grp in df_plot.groupby("bhk"):
                ax.scatter(grp["total_sqft"], grp["price"], alpha=0.25, s=8,
                           color=palette[bhk_val], label=f"{bhk_val} BHK")
            ax.set_xlabel("Total Square Feet"); ax.set_ylabel("Price (Lakh INR)")
            ax.set_title("sqft vs Price — colored by BHK")
            ax.legend(title="BHK", fontsize=8)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        elif page == "boxplot":
            top10 = df.groupby("location")["price"].median().nlargest(10).index
            df_top10 = df[df["location"].isin(top10)]
            order = df_top10.groupby("location")["price"].median().sort_values(ascending=False).index
            fig, ax = plt.subplots(figsize=(5.5, 3))
            sns.boxplot(data=df_top10, x="location", y="price", hue="location",
                        order=order, ax=ax, palette="Blues_r", legend=False,
                        flierprops={"marker": ".", "alpha": 0.4, "markersize": 3})
            ax.set_xlabel(""); ax.set_ylabel("Price (Lakh INR)")
            ax.set_title("Price Distribution — Top 10 Locations")
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        elif page == "actual_pred":
            fig, ax = plt.subplots(figsize=(3.5, 3.2))
            ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="#4C72B0")
            lims = [min(y_test.min(), y_pred.min()) - 5, max(y_test.max(), y_pred.max()) + 5]
            ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel("Actual (Lakh INR)"); ax.set_ylabel("Predicted (Lakh INR)")
            ax.set_title("Actual vs Predicted  (R²=0.78)")
            ax.legend(fontsize=8)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        elif page == "residuals":
            c1, c2 = st.columns(2)
            c1.metric("Mean Error", f"{mean_err:.2f} Lakh")
            c2.metric("MAE",        f"{mae:.2f} Lakh")
            fig, ax = plt.subplots(figsize=(5.5, 2.8))
            ax.hist(residuals, bins=55, color="#4C72B0", edgecolor="white", alpha=0.85)
            ax.axvline(0, color="red", linewidth=1.4, linestyle="--", label="Zero error")
            ax.axvline(mean_err, color="orange", linewidth=1.4,
                       label=f"Mean = {mean_err:.1f} Lakh")
            ax.set_xlabel("Actual − Predicted (Lakh INR)"); ax.set_ylabel("Count")
            ax.set_title("Distribution of Prediction Errors")
            ax.legend(fontsize=8)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Column 3: business Q + guideline ───
    with info_col:
        if page == "map":
            st.info("**Business Q**\nทำเลในพื้นที่ส่วนใดของบังกาลอร์มีระดับราคาสูงที่สุด และความหนาแน่นของธุรกรรมกระจายตัวอย่างไร?")
            st.markdown("""
**📖 วิธีอ่าน**
- วงใหญ่ = ราคา median สูง
- 🔴 แดง = แพง · 🔵 น้ำเงิน = ถูกกว่า
- Hover ดูชื่อ, median price, จำนวน Tx
            """)

        elif page == "scatter":
            st.info("**Business Q**\nพื้นที่และ BHK ส่งผลต่อราคาอย่างไร และตลาดมีกลุ่มสินค้าประเภทใดบ้าง?")
            st.markdown("""
**📖 วิธีอ่าน**
- แต่ละสีแทน BHK ต่างกัน
- กลุ่มสีแยกชัด = BHK เป็น factor สำคัญ
- แนวโน้มขึ้น = sqft มาก ราคาสูง
- จุดลอยสูง = outlier ทำเลพิเศษ
            """)

        elif page == "boxplot":
            st.info("**Business Q**\nทำเลไหนมีราคาสูงที่สุด และความผันผวนในแต่ละทำเลต่างกันอย่างไร?")
            st.markdown("""
**📖 วิธีอ่าน**
- เส้นกลาง box = median price
- box สูง = ราคาหลากหลาย
- box เตี้ย = ราคาสม่ำเสมอ
- จุดนอก whisker = Luxury outlier
            """)

        elif page == "actual_pred":
            st.info("**Business Q**\nโมเดลทำนายได้แม่นยำแค่ไหน และมีกลุ่มบ้านใดที่พลาดมาก?")
            st.markdown("""
**📖 วิธีอ่าน**
- จุดใกล้เส้นแดงประ = ทำนายแม่น
- เหนือเส้น = underestimate
- ใต้เส้น = overestimate
- R²=0.78 = อธิบาย 78% ของความแปรปรวน
            """)

        elif page == "residuals":
            st.info("**Business Q**\nโมเดลมี systematic bias ไหม และโดยเฉลี่ยคลาดเคลื่อนเท่าใด?")
            st.markdown("""
**📖 วิธีอ่าน**
- Residual = จริง − ทำนาย
- centered ที่ 0 = ไม่มี bias (ดี)
- หางขวายาว = บ้านแพง underestimate
- MAE = คลาดเคลื่อนเฉลี่ยไม่สนทิศทาง
            """)
