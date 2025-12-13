# MoodMate — Emotion-Aware NLP Microservice For Mental Wellness Support
MoodMate is an NLP-based microservice designed to detect user emotional states in short user-generated text and generate brief, empathetic support messages. The system demonstrates how Small Language Models (SLMs) can be effectively used to build emotion-aware applications under realistic computational and time constraints.

The project addresses the challenge of extracting emotional intent from natural language and responding in a contextually appropriate manner, using open-source instruction-tuned language models. MoodMate is implemented as a lightweight, fully local microservice, emphasizing efficiency, reproducibility, and privacy.

This project presents the design, implementation, and evaluation of an NLP-based system using small language models for emotion-aware text processing.

---

## 🎯 Problem Definition and Motivation

Understanding emotional cues in text is a fundamental problem in Natural Language Processing, with applications in mental wellness support, human–computer interaction, and affective computing. While large language models offer strong performance, they are often impractical due to computational cost and deployment constraints.

This project investigates the following question:

To what extent can small, open-source language models perform emotion classification and generate emotionally appropriate responses in a resource-constrained setting?

MoodMate explores this question by combining supervised emotion classification with conditional text generation, demonstrating that SLMs can deliver meaningful emotional awareness without reliance on large proprietary models.

---

## 🧩 System Overview

MoodMate is composed of two primary NLP components exposed through a RESTful API:

Emotion Classification
User input text is classified into one of seven predefined macro-emotions.

Support Message Generation
Based on the detected emotion, the system generates a short, empathetic response tailored to the emotional category.

The system follows a modular pipeline:

User Text → Emotion Classifier (SLM) → Emotion Label → Response Generator (SLM) → Support Message 

This separation allows independent evaluation and replacement of each NLP component.

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

This project fine-tunes **Qwen** (a decoder-only large language model) using **LoRA** (Low-Rank Adaptation) to classify text into the 7 macro-emotions defined during dataset preprocessing.

### 🔹 Model Used
- **Base Model:** Qwen 1.5 (1.8B or 4B)
- **Fine-Tuning Method:** LoRA
- **Task Type:** Instruction-style *causal language modeling*
- **Objective:** Train Qwen to generate the correct emotion label from a prompt.

**Prompt (input):**
Instruction: Identify the emotion of the following text. <br>
Text: I’m really stressed about tomorrow. <br>
Emotion: <br>

**Target (expected output):**
anxiety

The model learns to predict the correct emotion after the `"Emotion:"` token.

---

### 🧩 Why LoRA?

LoRA allows efficient fine-tuning by updating only ~1–2% of the model parameters.

Benefits:
- Fits on free Google Colab GPU (T4)
- Produces a small adapter file (150–200 MB)
- Leaves original Qwen weights unchanged
- Faster training and lower memory use

---

### 📊 Training Data Format

The processed dataset contains:

| text | labels |
|------|---------|
| "I'm overwhelmed with school." | anxiety |
| "This is amazing!" | joy |

During tokenization, **prompt + label are combined**:
Instruction: Identify the emotion of the following text. <br>
Text: <br>
Emotion: <br>

To prevent the model from learning the prompt itself, **all prompt tokens are masked with `-100`**, so the model only learns from the label tokens.

---

## ⚙️ Training Pipeline Overview

The fine-tuning workflow includes:

1. **Load processed dataset** (7 macro-emotion labels).
2. **Construct instruction-style prompts** for each example.
3. **Tokenize the combined prompt and label** into model input.
4. **Mask prompt tokens** using `-100` so only label tokens influence training.
5. **Apply LoRA adapters** to the base Qwen model.
6. **Fine-tune using HuggingFace Trainer**, updating only LoRA layers.
7. **Save the trained LoRA adapter** for use in the API.

---

### 🚀 Reproducible Training

Training can be reproduced using the notebook:
`training/train_qwen_lora.ipynb`

The notebook:
- Loads Qwen in 8-bit mode for memory efficiency  
- Applies LoRA configuration  
- Tokenizes dataset using the masking strategy  
- Runs fine-tuning  
- Saves LoRA weights to: qwen-emotion-lora/

You may then upload the adapter to HuggingFace Hub or place it in: `models/qwen_emotion/`

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




