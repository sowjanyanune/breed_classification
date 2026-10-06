# 🐶 CanineVision AI • Dog Breed Classification

An AI-powered dog breed classification application built with **Transfer Learning (MobileNetV2)**, **TensorFlow**, and a modern, interactive **Streamlit** user interface.

---

## ✨ Features & Highlights

- 🖼️ **Multi-Source Input**:
  - 📁 **File Upload**: Upload JPG, PNG, or WEBP dog photos.
  - 📸 **Live Webcam Snapshot**: Snap a photo directly from your device camera.
  - ✨ **Instant Sample Showcase**: 1-click testing with preloaded sample breeds (Golden Retriever, Siberian Husky, Pug, Beagle).
- 🧠 **Deep Learning Prediction**:
  - Real-time neural inference powered by fine-tuned MobileNetV2.
  - Top match prediction with confidence percentage meter and latency indicator.
  - 📊 **Top-4 Probability Distribution** breakdown.
- 🐕 **Rich Breed Insights & Profiles**:
  - Interactive trait badges & characteristic breakdown (**Size & Build**, **Temperament**, **Energy Level**, **Origin & Lifespan**).
  - Grooming & Care advice tailored to the identified breed.
  - 💡 Fun facts and trivia for each breed.
- 📖 **Interactive Breed Encyclopedia Sidebar**:
  - Browse and inspect information on all 10 supported dog breeds anytime.

---

## 🐕 Supported Dog Breeds

1. **Beagle** 🐶
2. **Chihuahua** 🐕
3. **English Setter** 🐕‍🦺
4. **Golden Retriever** 🦮
5. **Labrador Retriever** 🐕
6. **Miniature Poodle** 🐩
7. **Pug** 🐶
8. **Rottweiler** 🐕‍🦺
9. **Shih Tzu** 🐾
10. **Siberian Husky** 🐺

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/) with custom CSS & glassmorphism theme
- **Deep Learning**: [TensorFlow](https://www.tensorflow.org/) & Keras (MobileNetV2)
- **Image Processing**: Pillow (PIL) & NumPy

---

## 🚀 Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

3. Open your browser at `http://localhost:8501` to use the application.