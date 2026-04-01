import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import cv2
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ----------------------------------
# PAGE CONFIG (restores tab icon)
# ----------------------------------

st.set_page_config(
    page_title="Wind Turbine AI Inspection",
    page_icon="🌬️",
    layout="wide"
)

# ----------------------------------
# TITLE
# ----------------------------------

st.title("🌬️ Wind Turbine Blade AI Inspection System")
st.write("Upload turbine blade images to automatically detect structural damage using AI.")

# ----------------------------------
# LOAD MODEL
# ----------------------------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ----------------------------------
# PDF REPORT FUNCTION
# ----------------------------------

def generate_pdf(df, health_score, image_name):

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.setFont("Helvetica", 16)
    c.drawString(50, 750, "Wind Turbine Blade Inspection Report")

    c.setFont("Helvetica", 10)
    c.drawString(50, 720, f"Image: {image_name}")
    c.drawString(50, 700, f"Generated: {datetime.now()}")

    y = 660

    for i, row in df.iterrows():

        text = f"{row['Damage Type']} | Confidence: {row['Confidence']} | Severity: {row['Severity']}"

        c.drawString(50, y, text)

        y -= 20

    c.drawString(50, y-20, f"Turbine Health Score: {health_score}")

    c.save()

    buffer.seek(0)

    return buffer


# ----------------------------------
# IMAGE UPLOAD
# ----------------------------------

uploaded_files = st.file_uploader(
    "Upload Blade Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ----------------------------------
# PROCESS IMAGES
# ----------------------------------

if uploaded_files:

    for uploaded_file in uploaded_files:

        st.divider()

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption=uploaded_file.name)

        if st.button(f"Analyze {uploaded_file.name}"):

            with st.spinner("Running AI inspection..."):

                results = model.predict(image)

                r = results[0]

                plotted = r.plot()

                with col2:
                    st.image(plotted, caption="Detection Result")

                names = model.names

                report_data = []

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

                if len(df) > 0:

                    st.subheader("Inspection Report")

                    st.dataframe(df)

                    # ---------------------------
                    # DAMAGE DISTRIBUTION PIE
                    # ---------------------------

                    damage_counts = df["Damage Type"].value_counts().reset_index()
                    damage_counts.columns = ["Damage Type", "Count"]

                    fig1 = px.pie(
                        damage_counts,
                        values="Count",
                        names="Damage Type",
                        title="Damage Distribution"
                    )

                    st.plotly_chart(fig1, use_container_width=True)

                    # ---------------------------
                    # CONFIDENCE BAR CHART
                    # ---------------------------

                    fig2 = px.bar(
                        df,
                        x="Damage Type",
                        y="Confidence",
                        color="Severity",
                        title="Detection Confidence"
                    )

                    st.plotly_chart(fig2, use_container_width=True)

                # ---------------------------
                # TURBINE HEALTH GAUGE
                # ---------------------------

                fig3 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=health_score,
                    title={'text': "Turbine Health Score"},
                    gauge={
                        'axis': {'range':[0,100]},
                        'bar': {'color':"green"},
                        'steps':[
                            {'range':[0,40],'color':"red"},
                            {'range':[40,70],'color':"orange"},
                            {'range':[70,100],'color':"lightgreen"}
                        ]
                    }
                ))

                st.plotly_chart(fig3, use_container_width=True)

                # ---------------------------
                # AI RISK LEVEL
                # ---------------------------

                if health_score > 80:
                    risk = "Low Risk"
                elif health_score > 50:
                    risk = "Moderate Risk"
                else:
                    risk = "High Risk"

                st.metric("AI Risk Level", risk)

                # ---------------------------
                # DAMAGE HEATMAP
                # ---------------------------

                heatmap = r.plot()

                heatmap = cv2.applyColorMap(
                    cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY),
                    cv2.COLORMAP_JET
                )

                st.image(heatmap, caption="Damage Heatmap")

                # ---------------------------
                # CSV REPORT DOWNLOAD
                # ---------------------------

                if len(df) > 0:

                    csv = df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download CSV Report",
                        data=csv,
                        file_name="inspection_report.csv",
                        mime="text/csv"
                    )

                    # ---------------------------
                    # PDF REPORT DOWNLOAD
                    # ---------------------------

                    pdf_file = generate_pdf(df, health_score, uploaded_file.name)

                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_file,
                        file_name="inspection_report.pdf",
                        mime="application/pdf"
                    )

st.markdown("---")
st.caption("YOLOv8 Segmentation | Streamlit AI Monitoring Dashboard")

