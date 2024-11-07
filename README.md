# CyberGuard-AI: Cybercrime Text Classification Using Transformers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.46.2-green.svg)](https://huggingface.co/transformers/)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA%2012.1-brightgreen.svg)](https://developer.nvidia.com/cuda-toolkit)

An advanced Natural Language Processing system that classifies cybercrime descriptions using state-of-the-art transformer models. Built to help law enforcement and cybersecurity professionals quickly categorize and analyze cybercrime reports.

## 🔗 Quick Links and Downloads

### Essential Resources
- **GitHub Repository:** [CyberGuard-AI-Hackathon](https://github.com/mamoonjan7/CyberGuard-AI-Hackathon)
- **Pre-trained Model:** Download [final_model.pt](https://drive.google.com/file/d/17ebsgn96zfayUuRWjcZp0698EflxZuYF/view?usp=sharing)
- **Demo Video:** Watch [hakathon_Crime.mp4](https://drive.google.com/file/d/1cQKf88mvZJUl9FMVszNr0WOVnjyAZicR/view?usp=sharing)
- **Training Code:** View [Hakathon_crime.ipynb](https://github.com/mamoonjan7/CyberGuard-AI-Hackathon/blob/main/Hakathon_crime.ipynb)

## 📚 Table of Contents
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Support](#-support)

## ✨ Features

### Core Capabilities
- Real-time cybercrime text classification
- Multi-category prediction support
- Confidence scoring for predictions
- Batch processing capabilities
- Interactive command-line interface
- GPU-accelerated inference

### Technical Features
- DistilBERT-based transformer architecture
- Smart label encoding for unknown categories
- Efficient text preprocessing pipeline
- High accuracy classification results
- Comprehensive statistical analysis

## 💻 Requirements

### Hardware Requirements
- **CPU:** Intel/AMD 64-bit processor
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 2GB free space
- **GPU:** NVIDIA GPU with 4GB+ VRAM (tested on RTX 3050 Ti)

### Software Requirements
- **Operating System:** Windows 10/Linux/MacOS
- **Python:** 3.8.20 or higher
- **CUDA:** 12.1 (for GPU acceleration)

### Required Dependencies
```
torch==2.4.1
transformers==4.46.2
numpy==1.24.3
scikit-learn==1.3.0
sentence-transformers==3.2.1
tqdm==4.65.0
pandas==2.0.3
```

## 🛠️ Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/mamoonjan7/CyberGuard-AI-Hackathon.git
   cd CyberGuard-AI-Hackathon
   ```

2. **Set Up Python Environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Model**
   - Download [final_model.pt](https://drive.google.com/file/d/17ebsgn96zfayUuRWjcZp0698EflxZuYF/view?usp=sharing)
   - Place in project root directory

## 📊 Usage

### Interactive Command Line Interface
```bash
python interactive_predictor.py
```

The interface provides three options:
1. Test with custom input
2. Test with examples from test dataset
3. Exit

### Python API Usage
```python
from interactive_predictor import CybercrimePredictor

# Initialize predictor
predictor = CybercrimePredictor("final_model.pt")

# Make single prediction
text = "Received suspicious email asking for bank credentials"
category, confidence = predictor.predict(text)
print(f"Category: {category}")
print(f"Confidence: {confidence:.2%}")

# Batch processing example
texts = [
    "Unauthorized credit card transactions noticed",
    "Email claiming lottery win asking for details",
    "Ransomware encrypted all files"
]

for text in texts:
    category, confidence = predictor.predict(text)
    print(f"\nText: {text}")
    print(f"Category: {category}")
    print(f"Confidence: {confidence:.2%}")
```

## 🧠 Model Architecture

### Components
1. **Text Preprocessor:**
   - URL and email detection
   - Phone number identification
   - Special character handling
   - Efficient text cleaning

2. **Smart Label Encoder:**
   - Semantic similarity matching
   - Unknown category handling
   - Sentence transformer encoding

3. **Core Model:**
   - Base: DistilBERT
   - Custom classification head
   - GPU optimization

## 🔍 Troubleshooting

### Common Issues and Solutions

1. **Model Loading Errors**
   ```
   Check:
   - Model file exists in correct location
   - File permissions are correct
   - Sufficient disk space
   ```

2. **GPU Memory Issues**
   ```
   Solutions:
   - Close other GPU applications
   - Reduce batch size
   - Free GPU memory
   ```

3. **Dependency Conflicts**
   ```
   Solutions:
   - Use exact versions from requirements.txt
   - Create fresh virtual environment
   - Update CUDA if using GPU
   ```

## 📂 Project Structure
```
CyberGuard-AI-Hackathon/
├── interactive_predictor.py     # Main prediction interface
├── requirements.txt            # Package dependencies
├── Hakathon_crime.ipynb       # Training notebook
├── final_model.pt             # Pre-trained model
└── README.md                  # Documentation
```

## 👨‍💻 Creator
**Muhammad Mamoon jan**
- GitHub: [mamoonjan7](https://github.com/mamoonjan7)

## 🤝 Contributing
1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 🆘 Support
For assistance:
1. Create GitHub issue
2. Watch [demo video](https://drive.google.com/file/d/1cQKf88mvZJUl9FMVszNr0WOVnjyAZicR/view?usp=sharing)
3. Check troubleshooting guide

## 📝 License
MIT License

## ⚠️ Disclaimer
This tool is intended for legitimate cybersecurity research and analysis only. Use responsibly and in accordance with applicable laws and regulations.

## 🙏 Acknowledgments
- HuggingFace for transformers library
- PyTorch development team
- Cybersecurity community for datasets
