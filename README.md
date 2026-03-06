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

```
# Clone repository
git clone https://github.com/YOUR_USERNAME/synapse-prototype.git
cd synapse-prototype

# Create virtual environment
python -m venv synapse-env

# Activate virtual environment
# Windows:
.\synapse-env\Scripts\activate
# Linux/Mac:
source synapse-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: First run will download AI models (~2GB total). Requires internet connection initially.

### Run the System

```
python main.py
```

---

## 🎮 Controls

| Key | Action | Description |
|-----|--------|-------------|
| `W` | What's ahead? | Get summary of detected objects with audio |
| `D` | Describe scene | Generate natural language scene description |
| `R` | Read text | Extract and read text from current view |
| `T` | Toggle warnings | Enable/disable automatic trajectory warnings |
| `Q` | Quit | Exit system |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  User Interface                     │
│              (Voice Commands + Audio)               │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│           Intelligence & Control Layer              │
│  ┌──────────────┐      ┌─────────────────────────┐  │
│  │   Command    │      │  Adaptive Power         │  │
│  │  Processor   │      │  Manager (planned)      │  │
│  └──────────────┘      └─────────────────────────┘  │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              Processing Pipeline                    │
│  ┌─────────┐  ┌─────────┐  ┌──────┐  ┌──────────┐   │
│  │ YOLO +  │  │  BLIP   │  │ OCR  │  │   Face   │   │
│  │ Tracker │  │Caption. │  │      │  │ (planned)│   │
│  └─────────┘  └─────────┘  └──────┘  └──────────┘   │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│               Hardware Layer                        │
│         Camera | Mic | IMU (optional)               │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Performance Metrics

| Metric | Current (CPU) | Target (Pi + Optimization) |
|--------|--------------|----------------------------|
| Detection FPS | 9 FPS | 15-20 FPS |
| Tracking FPS | 4-5 FPS | 10-15 FPS |
| OCR Latency | 2-3s | <2s |
| Caption Latency | 3s | <2s |
| Total System Latency | <3s | <2s |
| Memory Usage | <3GB | <2GB |

---

## 🔬 Novel Contributions

### 1. Proactive Navigation
- First wearable system with trajectory-based collision prediction
- Shifts from reactive ("obstacle at 2m") to proactive ("person approaching from right")
- Demonstrated 17 directional events in testing

### 2. User-Centric Information Control
- Solves 7-year-old "information overload" problem (identified in 2018 HCI study)
- Voice-activated on-demand architecture
- User pulls information vs. system pushing continuously

### 3. Complete Multi-Modal Integration
- Only system combining detection, OCR, scene understanding, and social cues
- All processing on-device (privacy-first)
- Unified natural language interface

### 4. Context-Aware Power Management (Planned)
- IMU-based activity detection
- Dynamic model scheduling (stationary/walking/running modes)
- Target: 6+ hour battery life on Raspberry Pi 4

---

## 📂 Project Structure

```
synapse-prototype/
├── src/
│   ├── vision/
│   │   ├── tracker.py         # YOLOv11 + ByteTrack
│   │   ├── distance_estimator.py  # Monocular distance estimation
│   │   ├── spatial_layout.py  # Left/ahead/right zone detection
│   │   ├── scene_memory.py    # Short-term scene memory
│   │   ├── ocr.py             # EasyOCR wrapper
│   │   └── captioner.py       # BLIP wrapper
│   ├── io/
│   │   ├── camera.py          # Threaded camera capture
│   │   └── tts_output.py      # Text-to-speech engine
│   ├── utils/
│   │   ├── logger.py          # File + console logging
│   │   ├── async_processor.py # Background thread manager
│   │   └── fps_counter.py     # Performance monitoring
│   └── logic/
│       ├── output_generator.py # Natural language generation
│       └── command_processor.py # Command parsing (planned)
│    
├── tests/
│   ├── test_camera.py
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_ocr.py
│   └── test_caption.py
├── config.yaml                # Configuration file
├── main.py                    # Main orchestrator
├── requirements.txt           # Dependencies
├── README.md                  # This file\
├── logs/                      # Auto-generated session logs
└── assets/                    # Test images/videos
```

---

## 🧪 Testing

Individual module tests:

```
# Test camera input
python test_camera.py

# Test object detection
python test_detector.py

# Test tracking
python test_tracker.py

# Test OCR
python test_ocr.py

# Test scene captioning
python test_caption.py
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv11-Nano (Ultralytics) |
| Tracking | SORT / DeepSORT |
| OCR | EasyOCR + Tesseract |
| Scene Captioning | BLIP (Salesforce) |
| Face Recognition | DeepFace + MediaPipe (planned) |
| TTS | pyttsx3 (offline) |
| Wake Word | Porcupine (planned) |
| Vision Processing | OpenCV |
| Deep Learning | PyTorch, Transformers |

---

## 📈 Roadmap

### Phase 1: Core Prototype ✅ (Current)
- [x] Camera input with threading
- [x] Object detection (YOLO)
- [x] Basic tracking with direction prediction
- [x] OCR with TTS
- [x] Scene captioning
- [x] Unified interface

### Phase 2: Optimization (In Progress)
- [x] Improved tracking stability
- [x] Distance estimation (monocular, no depth sensor)
- [x] Spatial awareness (left/ahead/right zones)
- [x] Scene memory & change detection
- [x] Background threading (non-blocking OCR + captioning)
- [x] Frame skipping for CPU optimization
- [x] Session logging with FPS + CPU monitoring
- [ ] Model quantization (TFLite INT8)
- [ ] Raspberry Pi 4 deployment
- [ ] IMU integration for power management
- [ ] Wake-word activation

### Phase 3: Advanced Features
- [ ] Face recognition + emotion detection
- [ ] Depth estimation (monocular)
- [ ] Outdoor navigation mode
- [ ] Multi-language support

### Phase 4: Production
- [ ] User studies with visually impaired participants
- [ ] Wearable hardware enclosure design
- [ ] Battery optimization (6+ hours)
- [ ] Publication preparation

---

## 📚 Literature & References

Based on comprehensive survey of 20 papers (2007-2025):

**Key Papers:**
- YOLO-LITE (2018) - Embedded object detection
- SORT (2016) - Real-time tracking
- BLIP (2022) - Vision-language pre-training
- Information Overload Study (2018) - User-centric design

**Gaps Addressed:**
1. Single-modality focus → Multi-modal integration
2. Reactive navigation → Proactive trajectory prediction
3. Information overload → User-controlled interface
4. Cloud dependency → Complete on-device processing

See `docs/literature_survey.md` for full analysis.

---

## 👥 Team

- **Team Size**: 4 members
- **Institution**: [M S Ramaiah Institute of Technology]
- **Course**: Final Year Engineering Project
- **Supervisor**: [Dr. Sini Alex]

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- Ultralytics for YOLOv11
- Salesforce for BLIP model
- JaidedAI for EasyOCR
- HuggingFace for Transformers library
- Open-source community

---

## 📧 Contact

For questions or collaboration:
- **Email**: [shreyaspatil171@gmail.com]
- **Project Lead**: [Shreyas Patil]

---

## 📸 Demo

**Video**: [Link to demo video]

**Screenshots**:
- Live tracking with directional warnings
- Scene captioning output
- OCR text extraction

---

**Built with ❤️ for accessibility and inclusion**

*"Technology should empower everyone, regardless of ability."*

---

## 🐛 Known Issues

- Tracking ID stability needs improvement (optimization in Phase 2)
- OCR accuracy varies with lighting (70-85% currently)
- CPU performance bottleneck (will improve with Pi + quantization)

See [Issues](https://github.com/shreyas1017/synapse-prototype/issues) for tracking.