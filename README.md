# MoodMate — Emotion-Aware NLP Microservice For Mental Wellness Support
MoodMate is an NLP-based microservice designed to detect user emotions in short text and generate supportive, empathetic responses using small language models.  
This project demonstrates how emotion-aware systems can be built efficiently using open-source models such as Qwen and Phi.

---
## 📌 Features

- Emotion detection using **7 macro-emotions**  
- Supportive text generation using a small LLM  
- RESTful API built with FastAPI  
- Lightweight, privacy-friendly, and fully local  
- Dataset preprocessing pipeline included  
- Reproducible training and evaluation setup

---

## 📁 Repository Structure
moodmate/ <br>
├── api/                 `Contains scripts for FASTAPI` <br>
├── training/           `This folder contains all scripts related to training models.` <br>
├── evaluation/         `Contains all evaluation scripts` <br>
├── data/               `Holds both raw and processed datasets` <br>
│   ├── raw/            <br>
│   └── processed/      <br>
├── models/            <br>
├── docs/              `Holds all documentation material` <br>
├── notebooks/         `Notebooks for experimentation and exploration` <br>
├── README.md          <br>
└── requirements.txt   ## Defines the environment needed to run the project. <br>


---
## 📊 Dataset
This project uses the **GoEmotions dataset** (Google Research).  
The original dataset contains **28 fine-grained emotion labels**, which are mapped down to **7 macro-emotions** for improved accuracy and usability.

### **Macro-Emotion Mapping**
| Macro | Original Labels |
|-------|-----------------|
| joy | joy, excitement, amusement, pride |
| sadness | sadness, disappointment, grief |
| anger | anger, annoyance, frustration |
| anxiety | fear, nervousness, worry |
| love | love, caring, gratitude |
| surprise | surprise |
| neutral | neutral, confusion, curiosity, realization |

### **Processed Dataset**
The preprocessed dataset (7-emotion version) is located in: `/data/processed/goemotions_macro_7.csv` <br>
This allows users to reproduce experiments **without rerunning preprocessing**.
---

## 🧠 Models

### **Emotion Classifier**
- **Qwen 3–4B** fine-tuned using LoRA  
- Predicts one of the 7 macro-emotions

### **Support Message Generator**
- **Phi-3.5 Mini Instruct**  
- Generates short, empathetic responses conditioned on emotion

---

## ⚙️ Installation

> To be completed after API and environment setup.

Instructions will include:
- Creating virtual environment  
- Installing requirements  
- Setting Hugging Face access tokens  
- Running the API  

---

## 🚀 Usage

> To be completed after API development.

Will include:
- How to run the API  
- Sample requests to `/analyze_emotion`  
- Sample requests to `/generate_support`  

---

## 🏋️ Model Training

> To be completed after training scripts are developed.

This section will cover:
- Preparing training data  
- Running LoRA fine-tuning  
- Saving and loading models  

---

## 📈 Evaluation

> To be completed after evaluation scripts.

Metrics planned:
- Precision, Recall, F1 for classification  
- Human evaluation for generation  
- Latency and performance metrics  

---

## 📄 Results

> To be filled after model training and evaluation.

---

## 📚 Acknowledgements

- GoEmotions Dataset (Google Research)  
- Qwen & Phi models (Open-source)  
- HuggingFace Transformers library  

---




