import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time

st.set_page_config(
    page_title="Wind Turbine Damage Detection",
    page_icon="🌬️",
    layout="wide"
)

# ---------- HEADER ----------
st.title("🌬️ Wind Turbine Blade Damage Detection")
st.caption("AI powered structural inspection system")

st.markdown("---")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("runs/segment/train/weights/best.pt")

model = load_model()

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader(
    "Upload Wind Turbine Blade Image",
    type=["jpg","png","jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # LEFT SIDE
    with col1:

        st.subheader("Uploaded Image")
        st.image(image)

        analyze = st.button("🔍 Analyze Damage")

    # RIGHT SIDE
    with col2:

        if analyze:

            with st.spinner("Running AI structural inspection..."):

                time.sleep(1)

                results = model.predict(image)
                r = results[0]

                st.subheader("Detection Result")

                plotted = r.plot()
                st.image(plotted)

                st.markdown("## Structural Report")

                if len(r.boxes) == 0:

                    st.success("✅ No structural damage detected")

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

                    # ---------- REPORT TABLE ----------
                    st.dataframe(df, use_container_width=True)

                    # ---------- CONFIDENCE BARS ----------
                    st.markdown("### Confidence Levels")

                    for d in damages:
                        st.progress(d["Confidence"])
                        st.write(f'{d["Damage Type"]} ({d["Confidence"]})')

                    # ---------- PIE CHART ----------
                    st.markdown("### Damage Distribution")

                    damage_counts = df["Damage Type"].value_counts()

                    fig1, ax1 = plt.subplots()
                    ax1.pie(
                        damage_counts,
                        labels=damage_counts.index,
                        autopct='%1.1f%%',
                        startangle=90
                    )
                    ax1.axis("equal")

                    st.pyplot(fig1)

                    # ---------- CONFIDENCE CHART ----------
                    st.markdown("### Confidence Chart")

                    fig2, ax2 = plt.subplots()

                    ax2.bar(
                        df["Damage Type"],
                        df["Confidence"]
                    )

                    ax2.set_xlabel("Damage Type")
                    ax2.set_ylabel("Confidence")
                    ax2.set_title("Detection Confidence")

                    st.pyplot(fig2)

                    # ---------- DOWNLOAD REPORT ----------
                    csv = df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="📥 Download Inspection Report",
                        data=csv,
                        file_name="turbine_damage_report.csv",
                        mime="text/csv"
                    )

st.markdown("---")
st.caption("YOLOv8 Segmentation | Streamlit AI Inspection System")
