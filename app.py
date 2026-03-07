import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Wind Turbine AI Inspection",
    page_icon="🌬️",
    layout="wide"
)

# ---------- DARK STYLE ----------
st.markdown("""
<style>
body {
    background-color:#0E1117;
}
.report-card {
    background-color:#1f2937;
    padding:20px;
    border-radius:10px;
    margin-bottom:10px;
}
.metric-card {
    background-color:#1f2937;
    padding:15px;
    border-radius:10px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("🌬️ Wind Turbine Blade AI Inspection")
st.caption("Deep Learning Structural Damage Detection (YOLOv8)")

st.markdown("---")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("runs/segment/train/weights/best.pt")

model = load_model()

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader(
    "Upload Turbine Blade Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1,1])

    # ---------- LEFT PANEL ----------
    with col1:

        st.subheader("Input Image")
        st.image(image)

        analyze = st.button("Run AI Inspection")

    # ---------- RIGHT PANEL ----------
    with col2:

        if analyze:

            with st.spinner("Running structural analysis..."):

                results = model.predict(image)
                r = results[0]

                st.subheader("Detection Result")

                plotted = r.plot()
                st.image(plotted)

                if len(r.boxes) == 0:

                    st.success("No damage detected")

                else:

                    names = model.names
                    damages = []

                    for box in r.boxes:

                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        damage = names[cls]

                        damages.append({
                            "Damage Type": damage,
                            "Confidence": round(conf,2)
                        })

                    df = pd.DataFrame(damages)

                    # ---------- TURBINE HEALTH SCORE ----------
                    avg_conf = df["Confidence"].mean()

                    health_score = max(0, 100 - (avg_conf*100))

                    st.markdown("### Turbine Health Score")

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=health_score,
                        title={'text': "Health Index"},
                        gauge={
                            'axis': {'range': [0,100]},
                            'bar': {'color': "green"},
                            'steps': [
                                {'range': [0,40], 'color': "red"},
                                {'range': [40,70], 'color': "orange"},
                                {'range': [70,100], 'color': "green"}
                            ],
                        }
                    ))

                    st.plotly_chart(fig, use_container_width=True)

                    # ---------- SEVERITY ----------
                    st.markdown("### Damage Severity")

                    severity = "Low"

                    if avg_conf > 0.7:
                        severity = "High"
                    elif avg_conf > 0.4:
                        severity = "Medium"

                    st.info(f"Severity Level: **{severity}**")

                    # ---------- DAMAGE TABLE ----------
                    st.markdown("### Damage Report")

                    st.dataframe(df, use_container_width=True)

                    # ---------- PIE CHART ----------
                    st.markdown("### Damage Distribution")

                    fig2 = px.pie(
                        df,
                        names="Damage Type",
                        title="Damage Categories"
                    )

                    st.plotly_chart(fig2, use_container_width=True)

                    # ---------- CONFIDENCE BAR ----------
                    st.markdown("### Confidence Levels")

                    fig3 = px.bar(
                        df,
                        x="Damage Type",
                        y="Confidence",
                        color="Damage Type"
                    )

                    st.plotly_chart(fig3, use_container_width=True)

                    # ---------- DOWNLOAD REPORT ----------
                    csv = df.to_csv(index=False).encode()

                    st.download_button(
                        "Download Inspection Report",
                        csv,
                        "turbine_report.csv",
                        "text/csv"
                    )

st.markdown("---")
st.caption("YOLOv8 Segmentation | Streamlit AI Monitoring Dashboard")

