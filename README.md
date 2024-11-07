# CyberGuard-AI: Cybercrime Text Classification Using Transformers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.46.2-green.svg)](https://huggingface.co/transformers/)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA%2012.1-brightgreen.svg)](https://developer.nvidia.com/cuda-toolkit)

An advanced Natural Language Processing system that classifies cybercrime descriptions using state-of-the-art transformer models. Built to help law enforcement and cybersecurity professionals quickly categorize and analyze cybercrime reports.

## Quick Links
- [Project Repository](https://github.com/mamoonjan7/CyberGuard-AI-Hackathon)
- [Demo Video](https://drive.google.com/file/d/1cQKf88mvZJUl9FMVszNr0WOVnjyAZicR/view?usp=sharing)
- [Pre-trained Model](https://drive.google.com/file/d/17ebsgn96zfayUuRWjcZp0698eflxZuYF/view?usp=sharing)
- [Training Notebook](https://github.com/mamoonjan7/CyberGuard-AI-Hackathon/blob/main/Hakathon_crime.ipynb)

## Table of Contents
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [Support](#support)

## Features

### Core Capabilities
- Real-time cybercrime text classification
- Multi-category prediction support
- Confidence scoring for predictions
- Batch processing capabilities

### Technical Features
- DistilBERT-based transformer architecture
- Smart label encoding for unknown categories
- GPU-accelerated inference
- Efficient text preprocessing pipeline
- Interactive command-line interface

## System Requirements

### Hardware
- CPU: Intel/AMD 64-bit processor
- RAM: 8GB minimum (16GB recommended)
- Storage: 2GB free space
- GPU: NVIDIA GPU with 4GB+ VRAM (tested on RTX 3050 Ti)

### Software
- Operating System: Windows 10/Linux/MacOS
- Python: 3.8.20 or higher
- CUDA: 12.1 (for GPU acceleration)

### Dependencies
```
torch==2.4.1
transformers==4.46.2
numpy==1.24.3
scikit-learn==1.3.0
sentence-transformers==3.2.1
tqdm==4.65.0
pandas==2.0.3
```

## Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/mamoonjan7/CyberGuard-AI-Hackathon.git
   cd CyberGuard-AI-Hackathon
   ```

2. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model**
   - Get the pre-trained model from [Google Drive](https://drive.google.com/file/d/17ebsgn96zfayUuRWjcZp0698eflxZuYF/view?usp=sharing)
   - Place `final_model.pt` in the project root directory

## Usage

### Interactive Interface
```bash
python interactive_predictor.py
```

This launches the interactive CLI with options to:
1. Test individual text inputs
2. Process batch examples
3. View detailed statistics

### Python API
```python
from interactive_predictor import CybercrimePredictor

# Initialize
predictor = CybercrimePredictor("final_model.pt")

# Single prediction
text = "Suspicious email asking for bank credentials"
category, confidence = predictor.predict(text)
print(f"Category: {category}")
print(f"Confidence: {confidence:.2%}")

# Batch processing
texts = [
    "Unauthorized credit card charges detected",
    "Website asking for sensitive information",
    "Computer locked with ransom demand"
]

for text in texts:
    category, confidence = predictor.predict(text)
    print(f"\nText: {text}")
    print(f"Category: {category}")
    print(f"Confidence: {confidence:.2%}")
```

## Model Details

### Architecture
- Base Model: DistilBERT
- Custom classification head
- Smart label encoding system
- Efficient preprocessing pipeline

### Training
- Training details available in [Hakathon_crime.ipynb](https://github.com/mamoonjan7/CyberGuard-AI-Hackathon/blob/main/Hakathon_crime.ipynb)
- GPU-accelerated training process
- Optimized for cybercrime classification

### Performance
- Fast inference time with GPU acceleration
- Robust handling of various cybercrime categories
- Automatic handling of unseen categories

## Project Structure
```
CyberGuard-AI-Hackathon/
├── interactive_predictor.py
├── requirements.txt
├── Hakathon_crime.ipynb
├── final_model.pt
└── README.md
```

## Troubleshooting

### Common Issues
1. **Model Loading**
   ```
   Solution: Ensure model file is in correct location
   ```

2. **GPU Memory**
   ```
   Solution: Free GPU memory or reduce batch size
   ```

3. **Dependencies**
   ```
   Solution: Use exact versions from requirements.txt
   ```

## Contributing
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## Creator
**Muhammad Mamoon jan**
- GitHub: [mamoonjan7](https://github.com/mamoonjan7)

## Support
For issues or questions:
1. Create GitHub issue
2. Watch tutorial video
3. Check troubleshooting guide

## Disclaimer
This tool is for cybersecurity research and legitimate use only. Use responsibly and in accordance with applicable laws and regulations.

## License
MIT License

## Acknowledgments
- HuggingFace for transformers
- PyTorch development team
- Cybersecurity community
