# LLM Fine-Tuning PoC with QLoRA

A proof-of-concept implementation exploring Large Language Model fine-tuning using Quantized Low-Rank Adaptation (QLoRA). This project demonstrates efficient fine-tuning of Microsoft's Phi-2 model for dialogue summarization tasks.

## 🎯 Overview

This PoC showcases:
- Memory-efficient model fine-tuning using QLoRA
- Custom dataset processing for dialogue summarization
- Advanced quantization techniques
- Comprehensive evaluation metrics
- Optimized training pipeline

## 🛠 Technical Stack

### Core Components
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- bitsandbytes
- Hugging Face Datasets
- ROUGE evaluation metrics

### Model Architecture
- Base Model: Microsoft Phi-2
- Quantization: 4-bit NF4
- LoRA Configuration:
  - Rank: 16
  - Alpha: 16
  - Target Modules: Query, Key, Value projections and Dense layers
  - Dropout: 0.05

## 📊 Training Configuration

- Batch Size: 1
- Gradient Accumulation Steps: 4
- Learning Rate: 2e-4
- Optimizer: PagedAdamW8bit
- Training Steps: 1000
- Evaluation Frequency: Every 25 steps

## 🔍 Key Features

### Memory Optimization
- 4-bit quantization using bitsandbytes
- Gradient checkpointing
- Efficient memory management through QLoRA

### Training Pipeline
- Custom dataset preprocessing
- Instruction-tuning format
- Comprehensive logging system
- Automated evaluation metrics

### Evaluation System
- ROUGE metrics implementation
- Base vs. Fine-tuned model comparison
- Performance tracking

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the training script:
```bash
python main.py
```

## 📁 Project Structure

```
├── main.py              # Training and evaluation pipeline
├── requirements.txt     # Project dependencies
├── .gitignore          # Git ignore configuration
└── README.md           # Project documentation
```

## 📈 Future Improvements

- [ ] Multi-GPU training support
- [ ] Additional model architectures
- [ ] Enhanced evaluation metrics
- [ ] Web interface for model testing
- [ ] API endpoint implementation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📫 Contact

For questions and feedback, please open an issue in the repository.

---
Built with ❤️ using PyTorch and Hugging Face Transformers