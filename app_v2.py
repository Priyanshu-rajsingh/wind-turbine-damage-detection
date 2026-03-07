import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="Wind Turbine Damage Detection",
    layout="wide"
)

# ---------- HEADER ----------
st.markdown("""
# 🌬️ Wind Turbine Blade Damage Detection
AI powered structural inspection system
""")

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

    # LEFT SIDE (INPUT IMAGE)
    with col1:

        st.subheader("Uploaded Blade Image")
        st.image(image)

        analyze = st.button("Run Damage Analysis")

    # RIGHT SIDE (RESULTS)
    with col2:

        if analyze:

            with st.spinner("Analyzing blade structure..."):

                results = model.predict(image)

                r = results[0]

                st.subheader("Damage Detection Result")

                plotted = r.plot()
                st.image(plotted)

                st.markdown("### Structural Report")

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

                    st.dataframe(df)

                    st.markdown("### Confidence Levels")

                    for d in damages:
                        st.progress(d["Confidence"])
                        st.write(f'{d["Damage Type"]} ({d["Confidence"]})')

st.markdown("---")
st.caption("AI based turbine blade inspection | YOLOv8 segmentation")
