# 🧠 Medical Imaging Analyzer – AI-Powered Radiological Assistant

## 🔬 Project Description

**Medical Imaging Analyzer** is a smart AI-based radiology support system that integrates multiple **hybrid deep learning models** for the detection of critical diseases from medical images.  
This platform is designed to assist radiologists by automating the early diagnosis of conditions such as:

- 🫁 **Pneumonia** – detected from chest X-rays using CNN-ViT hybrid architectures.
- 🧠 **Brain Tumors** – identified from MRI scans with enhanced accuracy using GAN-augmented CNN models.

The system aims to provide real-time, high-confidence predictions by combining convolutional neural networks (CNNs), vision transformers (ViTs), and data augmentation techniques like GANs. These methods improve model generalization, making the system highly reliable across varied patient data.

The end goal is to develop a **comprehensive AI-powered radiological assistant** that supports multi-disease detection and improves diagnostic speed, accuracy, and explainability.

---

## 🧪 Key Capabilities

- ⚙️ Uses hybrid models (CNN + ViT / GAN-CNN) for robust detection
- 📷 Accepts and processes CT, X-ray, and MRI scans
- 💡 Designed for future integration of TB, liver lesions, and other abnormalities
- 🔍 In-progress support for Grad-CAM/SHAP to visualize decision-making
- 💬 Clear web interface for real-time analysis and user feedback

---

## 💻 Tech Stack

- Python (Flask)
- TensorFlow / PyTorch
- HTML, CSS, JavaScript
- OpenCV, NumPy, Matplotlib
- Grad-CAM / SHAP (planned for explainability)

---

## 📁 Directory Overview

├── app.py # Main Flask app
├── static/ # Assets like CSS, JS, uploaded images
├── templates/ # HTML files for UI
├── model/ # Trained AI models (.h5, .pt)
├── utils/ # Image preprocessing, helpers
├── README.md # Project documentation

yaml
Copy
Edit

---

## 🛠️ Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/HaswinAI/Medical_imaging_analyzer.git
   cd Medical_imaging_analyzer
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the application

bash
Copy
Edit
python app.py
Then visit: http://127.0.0.1:5000 in your browser.

📌 To-Do & Roadmap
 Cancer detection from CT and X-ray

 Multi-disease classification (liver, pneumonia, tumors)

 SHAP / Grad-CAM visual explanations

 Multi-language support

 Deployment on cloud (Render/Heroku)

🤝 Contributing
We welcome contributions! Feel free to fork the repo, submit issues, or open pull requests.

📧 Contact
For questions or collaborations:
📩 haswin.ai@gmail.com

📄 License
This project is licensed under the MIT License.
Feel free to use, modify, and distribute with attribution.

yaml
Copy
Edit

---
