# DFAT: Dual-stage Fusion of Acoustic and Text Feature for Speech Emotion Recognition

This repository contains the official implementation of the paper:

**DFAT: Dual-stage Fusion of Acoustic and Text Feature for Speech Emotion Recognition (VLSP 2025)**

## 📌 Overview
DFAT is a lightweight **dual-stage fusion pipeline** for Speech Emotion Recognition (SER).  
It combines:
- **Acoustic features** from pretrained encoders (e.g., Emotion2Vec, WavLM).  
- **Text features** from ASR transcripts using Whisper-small.  
- **Hybrid fusion** with Logistic Regression, Random Forest, and XGBoost, aggregated by an Optuna-tuned ensemble.  

On the VLSP 2025 private test set, DFAT achieves **0.8438 WA**, ranking **2nd place** in the competition.

<p align="center">
  <img src="images/Pipeline.png" width="600"/>
</p>

## 📊 Datasets
We use three publicly available Vietnamese datasets:
- [ViSEC](https://huggingface.co/datasets/hustep-lab/ViSEC)  
- [28k-vn](https://huggingface.co/datasets/natmin322/28k_vietnamese_voice_augmented_of_VigBigData)  
- [PhoAudiobook](https://huggingface.co/datasets/thivux/phoaudiobook)  

## 🚀 Usage
Code and training scripts will be released soon.  

## 📬 Contact
For questions and collaborations, please reach out to:  
- 📧 nhitny2802@gmail.com  
- 📧 lequangsang@siu.edu.vn  
- 📧 tranquanghuyk15@siu.edu.vn  

