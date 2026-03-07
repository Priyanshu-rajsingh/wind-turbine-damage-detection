

---

# Wind Turbine Blade Damage Detection

A deep learning–based system for automated detection of structural damage in wind turbine blades using YOLOv8 segmentation. The system analyzes blade images to identify defects such as erosion, contamination, laminate damage, and other surface anomalies, and delivers results through an interactive Streamlit web interface.

🔗 **Live Demo:** [wind-turbine-damage-detection.streamlit.app](https://wind-turbine-damage-detection.streamlit.app/)

---

## Features

- YOLOv8 segmentation for precise defect localization
- Automated structural inspection report generation
- Interactive web interface for image upload and analysis
- Visualization of detection results and damage metrics
- CSV and PDF inspection report export

---

## Project Structure

```
wind-turbine-damage-detection/
├── app.py               # Streamlit web application
├── dataset/             # Training dataset (images and labels)
├── configs/             # Dataset configuration files
├── src/                 # Training and prediction scripts
├── runs/                # Model training outputs and weights
├── requirements.txt     # Project dependencies
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Priyanshu-rajsingh/wind-turbine-damage-detection.git
cd wind-turbine-damage-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

---

## Dataset

The model was trained on a labeled dataset of turbine blade images with segmentation annotations across multiple damage categories.

---

## Technologies

| Tool | Purpose |
|------|---------|
| Python | Core language |
| YOLOv8 (Ultralytics) | Object detection & segmentation |
| PyTorch | Deep learning framework |
| OpenCV | Image processing |
| Streamlit | Web interface |
| Plotly | Data visualization |

---

## Applications

- Automated wind turbine inspection
- Structural health monitoring
- Predictive maintenance in renewable energy systems

---

## Author

**Priyanshu Raj Singh**  
Computer Science Engineering, Vardhaman College of Engineering
