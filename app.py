import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import json
import datetime

# 1. Page Config
st.set_page_config(page_title="🐟 Fish Classifier Dashboard", layout="wide")

# 2. Model + Labels Load
MODEL_PATH = r"E:\vscode\project5\models\mobilenetv2_fish_finetuned.h5"
LABELS_PATH = r"E:\vscode\project5\data\train"

model = load_model(MODEL_PATH)
class_labels = sorted(os.listdir(LABELS_PATH))

# 3. Sidebar
st.sidebar.image("https://i.imgur.com/kpE6zGW.png", use_column_width=True)
st.sidebar.title("🎯 Model Info")
st.sidebar.markdown("""
- **Model**: MobileNetV2 (fine-tuned)  
- **Input Size**: 224x224  
- **Classes**: {n}
- **Confidence Threshold**: Customize below
""".format(n=len(class_labels)))

confidence_threshold = st.sidebar.slider("🔧 Filter chart (min confidence %)", 0, 100, 1, step=1)

# 4. Main Title
st.title("🐠 Fish Species Classifier Dashboard")
st.markdown("Upload a fish image to identify its species with AI. See predictions, confidence scores, and a beautiful summary.")

# 5. File Upload
uploaded_file = st.file_uploader("📤 Upload a fish image", type=["jpg", "jpeg", "png"])

# 6. If Image Uploaded
if uploaded_file is not None:
    try:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)



        with col2:
            # Preprocessing
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            with st.spinner("🔍 Analyzing..."):
                prediction = model.predict(img_array)
                predicted_index = np.argmax(prediction)
                predicted_label = class_labels[predicted_index]
                confidence_score = prediction[0][predicted_index] * 100

            # Highlight Prediction
            st.success(f"### ✅ Predicted Species: **{predicted_label}**")
            st.metric(label="🎯 Confidence", value=f"{confidence_score:.2f}%")

            # Top 3 Predictions
            st.subheader("📌 Top 3 Likely Species")
            top_indices = prediction[0].argsort()[-3:][::-1]
            for i in top_indices:
                st.write(f"🔸 **{class_labels[i]}** — `{prediction[0][i]*100:.2f}%`")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        st.stop()

    # 7. Chart - Filtered
    st.subheader("📊 Full Class Confidence Distribution")
    filtered_indices = [i for i, score in enumerate(prediction[0]) if score * 100 >= confidence_threshold]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(
        [class_labels[i] for i in filtered_indices],
        [prediction[0][i] * 100 for i in filtered_indices],
        color="mediumturquoise"
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Confidence (%)")
    plt.title("AI Confidence per Class (Filtered)")
    st.pyplot(fig)

    # 8. Download Prediction
    result_data = {
        "filename": uploaded_file.name,
        "predicted": predicted_label,
        "confidence": float(round(confidence_score, 2)),
        "top_3": {
            class_labels[i]: float(round(prediction[0][i] * 100, 2))
            for i in top_indices
        },
        "timestamp": str(datetime.datetime.now())
    }

    st.download_button(
        label="📥 Download Prediction (JSON)",
        data=json.dumps(result_data, indent=2),
        file_name=f"{predicted_label}_prediction.json",
        mime="application/json"
    )

else:
    st.info("👈 Upload a fish image using the uploader to start classification.")

