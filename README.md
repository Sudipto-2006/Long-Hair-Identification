# ✂️ Age-Gated Hair Length & Gender Classification System

A Streamlit web application that couples computer vision feature extraction with a custom decision-engine pipeline. The system enforces strict hair-length classification rules for individuals aged **20 to 30**, while preserving standard baseline gender predictions outside this target age bracket.

---

## 📌 Problem Overview

Standard demographic classification models predict attributes independently. This project applies a conditional business logic layer on top of extracted image features:

- **Target Demographic ($20 \le \text{Age} \le 30$):**
  - **Long Hair** $\rightarrow$ Classified as **Female** (regardless of biological sex).
  - **Short Hair** $\rightarrow$ Classified as **Male** (regardless of biological sex).
- **Non-Target Demographic ($\text{Age} < 20$ or $\text{Age} > 30$):**
  - Standard baseline gender prediction is retained regardless of hair length.

---

## 🧠 Decision Matrix

| Detected Age Bracket | Hair Length | Raw / Baseline Gender | Final Model Output | Applied Logic Category |
| :--- | :--- | :--- | :--- | :--- |
| **$20 \le \text{Age} \le 30$** | **Long** | Male | **Female** | Inversion / Forced Female |
| **$20 \le \text{Age} \le 30$** | **Long** | Female | **Female** | Rule Concordant |
| **$20 \le \text{Age} \le 30$** | **Short** | Male | **Male** | Rule Concordant |
| **$20 \le \text{Age} \le 30$** | **Short** | Female | **Male** | Inversion / Forced Male |
| **$\text{Age} < 20$ or $\text{Age} > 30$** | Any | Male | **Male** | Default Preservation |
| **$\text{Age} < 20$ or $\text{Age} > 30$** | Any | Female | **Female** | Default Preservation |

---

## ✨ Features

- **Interactive GUI:** Built with Streamlit for real-time inference and calibration.
- **Dual Operating Modes:**
  - **Image Upload Mode:** Analyzes uploaded portraits, extracts face bounding boxes (ROI), and measures shoulder-region edge densities to estimate hair length.
  - **Logic Simulator Mode:** Allows manual slider/radio adjustments to test boundary conditions ($19 \leftrightarrow 20$ and $30 \leftrightarrow 31$) instantly.
- **Fail-Safe Computer Vision Pipeline:** Features an automated fallback ROI generator to prevent execution crashes in headless or minimal OpenCV environments.
- **Clear Diagnostic Output:** Visual indicators highlight when a rule override/inversion has occurred.

---

## 📂 Project Structure

```text
├── app.py              # Main Streamlit application and inference pipeline
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
