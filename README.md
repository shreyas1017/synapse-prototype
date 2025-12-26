# Project SYNAPSE 🧠

**A Real-Time Embedded Vision System for Assistive Navigation and Contextual Awareness**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Prototype-orange.svg)

---

## 📋 Overview

SYNAPSE is an intelligent, wearable assistive device designed for visually impaired individuals. It provides real-time environmental understanding through computer vision and AI, offering:

- **Proactive Navigation** - Trajectory-based warnings for approaching obstacles
- **Scene Understanding** - Natural language descriptions of surroundings
- **Text Recognition** - OCR for signs, labels, and documents
- **Social Awareness** - Face detection and emotion recognition (planned)
- **Interactive Control** - User-driven, on-demand information retrieval

---

## 🎯 Key Features

### 1. Object Detection & Tracking
- YOLOv11-Nano for real-time object detection
- SORT-based tracking with stable IDs
- **Novel**: Trajectory prediction for proactive warnings

### 2. Optical Character Recognition
- EasyOCR for "text-in-the-wild" extraction
- Preprocessing pipeline for enhanced accuracy
- Text-to-speech output

### 3. Scene Captioning
- BLIP (Salesforce) for natural language scene descriptions
- Context-aware formatting
- <3 second latency on CPU

### 4. Interactive Voice Interface
- On-demand module activation
- **Novel**: Solves information overload problem
- Natural language output

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Webcam
- Windows/Linux/MacOS
- 8GB RAM minimum

### Installation

