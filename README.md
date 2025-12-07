# 🧠 Alzheimer’s Disease Classification using CNN

A deep-learning based image classification system that predicts the stage of Alzheimer’s disease from MRI images. The model analyzes brain scans and classifies them into **four stages of cognitive decline**, helping support early detection and clinical assessment.

---

##  Overview

Alzheimer’s disease is a progressive neurodegenerative disorder that affects memory, thinking, and behavior. It is the most common cause of dementia worldwide. Early detection is crucial because timely diagnosis enables better care planning, treatment, and quality of life improvements.

This project uses a **Convolutional Neural Network (CNN)** trained on MRI images to classify Alzheimer’s disease into **four categories**, each representing a different stage of cognitive decline.

---

##  Classification Categories

### **1️⃣ Non-Demented**
Individuals show no signs of cognitive impairment.  
Memory, reasoning, and thinking abilities are normal.

---

### **2️⃣ Very Mild Demented**
Early signs of cognitive decline begin to appear.  
People may have minor memory lapses (e.g., forgetting names) but can still perform daily tasks independently.

---

### **3️⃣ Mild Demented**
Cognitive issues are more noticeable.  
This stage may include:
- Problems with memory and attention  
- Difficulty managing finances  
- Trouble following conversations  
- Need for partial assistance in daily activities  

---

### **4️⃣ Moderate Demented**
Significant cognitive decline that impacts everyday life.  
Individuals may:
- Forget personal information  
- Fail to recognize family members  
- Show reduced speech/understanding  
- Experience confusion or behavioral changes  

---

<table>
  <tr>
    <td align="center">
      <img src="images/mild.jpg" width="300px" /><br>
      <b>Mild Demented</b>
    </td>
    <td align="center">
      <img src="images/mod.jpg" width="300px" /><br>
      <b>Moderate Demented</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/veryMild.jpg" width="300px" /><br>
      <b>Very Mild Demented</b>
    </td>
    <td align="center">
      <img src="images/non.jpg" width="300px" /><br>
      <b>Non Demented</b>
    </td>
  </tr>
</table>


---

##  Model Details

- **Model Type:** Custom Convolutional Neural Network (CNN)  
- **Input Image Size:** 128 × 128 × 3  
- **Framework:** TensorFlow / Keras  
- **Loss Function:** Categorical Cross-Entropy  
- **Activation:** Softmax for final classification  
- **Output:** 4 Alzheimer’s disease stages  
- **File:** `cnn_model.h5`  

---

## 🎯 Features

-  **MRI Image Upload** → Predicts Alzheimer’s class  
-  **Real-time classification with confidence score**  
-  **FastAPI backend for quick inference**  
-  **Clean and structured project layout**  
-  **Accurate predictions using a trained CNN model**  


---

## 🧠 How the Model Works

1. The uploaded MRI image is resized to **128×128**.  
2. Pixel values are normalized (0–1).  
3. The CNN extracts spatial and structural patterns in the brain MRI.  
4. A dense layer outputs probabilities for each of the four classes.  
5. The highest-probability class = predicted stage.

---

##  How to Run the Project Locally

### **1. Create Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate
```
### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **3. Start the Server**
```bash
uvicorn app:app --reload
```

### **Server runs at:**
**http://127.0.0.1:8000**

---
## Output Snapshots

<div style="display: flex; height: 50;">
  <img src="images/op2.png" style="height: 50%; width: auto;">
  <img src="images/op1.png" style="height: 50%; width: auto;">
</div>



---

## 🌍 Real-World Applications

-  **Clinical Decision Support:**  
  Helps radiologists identify early signs of Alzheimer’s disease through automated MRI scan analysis.

-  **Screening Tool:**  
  Supports large-scale MRI screening programs for early detection in at-risk populations.

-  **Medical Research:**  
  Useful for studying structural brain changes across different dementia stages.

-  **Telemedicine:**  
  Enables remote diagnosis by allowing MRI scans to be uploaded and analyzed from anywhere.


---

##  Future Scope

-  **Cloud Deployment:**  
  Deploy the model on cloud platforms like AWS, Azure, or GCP for real-world accessibility and scalability.

-  **Cross-Platform Front-End:**  
  Build a React-based web UI or Android mobile app for seamless interaction and real-time predictions.

-  **Advanced Model Architectures:**  
  Upgrade to transfer learning models such as **VGG16**, **ResNet50**, **Inception**, or **EfficientNet** for higher accuracy.

-  **3D MRI Support:**  
  Extend the system to handle 3D MRI volumes using **3D Convolutional Neural Networks**.

-  **Explainable AI (XAI):**  
  Use visualization techniques like **Grad-CAM** to highlight brain regions influencing the model’s predictions, improving interpretability.

