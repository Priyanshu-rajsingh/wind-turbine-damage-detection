import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Wind Turbine Blade Damage Detection")

st.write("Upload a blade image to generate a structural damage report.")

# load model safely
@st.cache_resource
def load_model():
    return YOLO("runs/segment/train/weights/best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    #st.image(image, caption="Uploaded Image", use_container_width=True)
    st.image(image, caption="Uploaded Image")

    if st.button("Analyze Damage"):

        with st.spinner("Analyzing image..."):

            results = model.predict(image)

            r = results[0]

            st.subheader("Structural Report")

            if len(r.boxes) == 0:
                st.success("No structural damage detected.")

            else:

                names = model.names

                for box in r.boxes:

                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    damage = names[cls]

                    st.write(f"Damage Type: **{damage}**")
                    st.write(f"Confidence: **{conf:.2f}**")

            plotted = r.plot()

            # st.image(plotted, caption="Detection Result", use_container_width=True)
            st.image(plotted, caption="Detection Result")



