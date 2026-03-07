import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Wind Turbine Damage Detection",
    layout="wide"
)

# -----------------------------------
# TITLE
# -----------------------------------

st.title("🌬️ Wind Turbine Blade AI Inspection System")
st.write("Upload a turbine blade image to analyze structural damage using AI.")

# -----------------------------------
# LOAD MODEL
# -----------------------------------

@st.cache_resource
def load_model():
    return YOLO("runs/segment/train/weights/best.pt")

model = load_model()

# -----------------------------------
# IMAGE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload Blade Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# IF IMAGE PROVIDED
# -----------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image")

    if st.button("🔍 Analyze Damage"):

        with st.spinner("Running AI Inspection..."):

            results = model.predict(image)

            r = results[0]

            plotted = r.plot()

            with col2:
                st.image(plotted, caption="Detection Result")

            st.subheader("📊 Structural Inspection Report")

            names = model.names

            report_data = []

            # -------------------------
            # NO DAMAGE
            # -------------------------

            if len(r.boxes) == 0:

                st.success("No structural damage detected.")

                health_score = 100

            else:

                health_score = max(0, 100 - (len(r.boxes) * 15))

                for i, box in enumerate(r.boxes):

                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    damage = names[cls]

                    if conf > 0.75:
                        severity = "High"
                    elif conf > 0.45:
                        severity = "Medium"
                    else:
                        severity = "Low"

                    report_data.append({
                        "Image": uploaded_file.name,
                        "Detection ID": i+1,
                        "Damage Type": damage,
                        "Confidence": round(conf,2),
                        "Severity": severity,
                        "Health Score": health_score,
                        "Timestamp": datetime.now()
                    })

                df = pd.DataFrame(report_data)

                st.dataframe(df)

                # -------------------------
                # DAMAGE PIE CHART
                # -------------------------

                damage_counts = df["Damage Type"].value_counts().reset_index()
                damage_counts.columns = ["Damage Type","Count"]

                fig1 = px.pie(
                    damage_counts,
                    values="Count",
                    names="Damage Type",
                    title="Damage Distribution"
                )

                st.plotly_chart(fig1, use_container_width=True)

                # -------------------------
                # CONFIDENCE BAR CHART
                # -------------------------

                fig2 = px.bar(
                    df,
                    x="Damage Type",
                    y="Confidence",
                    color="Severity",
                    title="Detection Confidence Levels"
                )

                st.plotly_chart(fig2, use_container_width=True)

            # --------------------------------
            # TURBINE HEALTH GAUGE
            # --------------------------------

            fig3 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health_score,
                title={'text': "Turbine Health Score"},
                gauge={
                    'axis': {'range': [0,100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range':[0,40],'color':"red"},
                        {'range':[40,70],'color':"orange"},
                        {'range':[70,100],'color':"lightgreen"}
                    ],
                }
            ))

            st.plotly_chart(fig3, use_container_width=True)

            # --------------------------------
            # DOWNLOAD REPORT
            # --------------------------------

            if len(report_data) > 0:

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Download Inspection Report",
                    data=csv,
                    file_name="turbine_damage_report.csv",
                    mime="text/csv"
                )

st.markdown("---")
st.caption("YOLOv8 Segmentation | Streamlit AI Monitoring Dashboard")

