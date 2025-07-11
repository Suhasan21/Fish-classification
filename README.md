🐟 Multiclass Fish Image Classification using Transfer Learning 📌 Project Overview This project aims to classify different species of fish using deep learning and transfer learning techniques. Users can upload fish images through a Streamlit web app, and the model will predict the fish species along with confidence scores.

📁 Dataset Dataset contains fish images organized into folders by class.

It includes 11 fish categories.

The data is divided into:

train/ → Training images

val/ → Validation images

✅ Source: [Add dataset source or URL here if available]

🧠 Models Used CNN (from scratch)

VGG16

ResNet50

MobileNetV2 ✅ (Best Model)

InceptionV3

EfficientNetB0

All pretrained models were implemented using transfer learning. Convolution layers were frozen, and custom dense layers were added and trained.

🏆 Model Evaluation Evaluation metrics:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

MobileNetV2 achieved the highest performance and was selected for deployment.

🚀 Streamlit Web App Users can upload fish images

The model will:

Predict the fish species

Display confidence scores

Deployed using streamlit run app.py

🛠 Project Structure kotlin Copy Edit project/ ├── data/ │ ├── train/ │ └── val/ ├── models/ │ └── mobilenetv2_fish_best.h5 ├── app.py ├── train_models.py ├── evaluate_models.py ├── requirements.txt └── README.md 📦 How to Run Clone this repository:

bash Copy Edit git clone https://github.com/yourusername/fish-classification.git cd fish-classification Install dependencies:

bash Copy Edit pip install -r requirements.txt Run the Streamlit app:

bash Copy Edit streamlit run app.py 🧪 Libraries Used TensorFlow / Keras

Streamlit

NumPy

Pandas

Scikit-learn

Matplotlib / Seaborn
