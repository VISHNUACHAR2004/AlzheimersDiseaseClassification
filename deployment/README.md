# Alzheimer’s Stage Classification – Cloud Deployment (AWS FastAPI)

## 📌 Overview
This is a cloud-based Alzheimer’s Classification API deployed on Amazon Web Services (AWS).  
A trained Convolutional Neural Network (CNN) model predicts the stage of Alzheimer’s disease from MRI images.

The API is built using **FastAPI** and deployed on an **AWS EC2 instance** (Ubuntu 22.04).

## 🚀 Features
- Predicts 4 Alzheimer’s stages:
  - Mild Demented
  - Moderate Demented
  - Non-Demented
  - Very Mild Demented
- FastAPI-based REST API
- Accepts uploaded MRI images (.jpg, .png)
- Returns prediction + confidence score
- Secure, lightweight deployment
- Pure cloud-based inference (AWS)

## 🧠 Tech Stack
- Python 3
- FastAPI
- TensorFlow CPU
- Uvicorn
- AWS EC2

## 📁 Project Structure



Open:  
`http://127.0.0.1:8000/docs`

## ☁️ AWS Deployment Steps (Summary)
1. Launch EC2 instance (Ubuntu)
2. Install Python + pip
3. Install project dependencies
4. Upload project files to EC2
5. Run the FastAPI server
6. Access public prediction API via EC2 public IP

---

## 📸 Screenshots to Include in Assignment
- EC2 instance dashboard
- Security group inbound rules
- Terminal showing `uvicorn` running on EC2
- `/docs` Swagger UI running on EC2 IP
- Prediction response JSON

---

## 📡 API Endpoint
### POST `/predict`
Upload image → Get prediction + confidence.

---

## 🏁 Conclusion
This project demonstrates deploying an ML model to the cloud using AWS EC2 and FastAPI.  
It fulfills the core concepts of **cloud computing**, such as compute provisioning, API hosting, remote inference, and scalable architecture.

