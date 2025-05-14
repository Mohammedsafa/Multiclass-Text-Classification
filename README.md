# 🧠 Multiclass Text Classification with Transformers and Pretrained Embeddings

A deep learning project for classifying Arabic/English news articles into five categories: **Politics**, **Sports**, **Media**, **Market & Economy**, and **STEM**. This project investigates the impact of different embedding techniques, model architectures, and regularization strategies on classification performance.

---

## 📌 Table of Contents

- [📌 Table of Contents](#-table-of-contents)
- [📂 Project Overview](#-project-overview)
- [🛠️ Model Architectures](#️-model-architectures)
- [🧪 Experiments & Results](#-experiments--results)
- [🎯 Regularization Techniques](#-regularization-techniques)
- [🚀 Inference Example](#-inference-example)
- [📊 Evaluation](#-evaluation)
- [👨‍💻 Author](#-author)
- [🧾 Project Context](#-project-context)
- [📚 References and Acknowledgments](#-references-and-acknowledgments)

---

## 📂 Project Overview

We explored various embeddings and architectures for text classification. Our primary goal was to improve validation accuracy while keeping the model efficient and generalizable.

---

## 🛠️ Model Architectures

### 1. ✅ Final Model (Best Performance)

RoBERTa Embedding → Dropout → Transformer Encoder → BiGRU → Fully Connected → Softmax


- Pretrained RoBERTa embeddings (frozen)
- Transformer encoder layer with GELU and LayerNorm
- Bidirectional GRU layer for temporal learning
- Fully connected output layer for classification

---

## 🧪 Experiments & Results

| Embedding Type         | Model Architecture          | Validation Accuracy |
|------------------------|-----------------------------|---------------------|
| Trainable Embedding    | LSTM + Fully Connected      | 70%                 |
| GloVe Pretrained       | Transformer + FC Layers     | 74.25%              |
| **RoBERTa Pretrained** | Transformer Encoder + BiGRU | **77.15%**          |

---

## 🎯 Regularization Techniques

To prevent overfitting and improve generalization:

- **Label Smoothing**: Makes model less confident, improves generalization.
- **Class Weights**: Balances loss impact of underrepresented classes.
- **Dropout**: Applied after embedding layer.
- **Early Stopping**: Stops training when validation acc stops improving.

> 📌 **Note**: Training loss is **higher than validation loss** due to regularization methods.

---

## 🚀 Inference Example

```python
text = "Managing cash flow effectively is crucial for any business, ensuring that expenses donÃ¢â,¬â„¢t exceed income."
input_ids, mask = prepare_single_input(text, tokenizer, max_len=120)
output = model(input_ids, mask)
pred = torch.argmax(output, dim=1)
print("Predicted Class:", label_map[int(pred.item())])
The prediction class is:  Market & Economy
```

## 📊 Evaluation

 - Metrics: Accuracy, Macro F1-score, Confusion Matrix

 - Visualized using Seaborn heatmaps

 - Inference tested on unseen examples

## 👨‍💻 Author
Developed by Mohammed Safa for academic and purposes.


## 🧾 Project Context
This project was initially developed as part of a university semester course. After the course ended, additional enhancements were made to improve performance and generalization. It serves as a practical exploration of embedding techniques, model design, and training strategies for natural language classification.

## 📚 References & Acknowledgments
We acknowledge the use of the following tools and resources:

 - RoBERTa pretrained models from HuggingFace 🤗 Transformers.

 - GloVe embeddings from the Stanford NLP Group.

 - HuggingFace and PyTorch documentation.

 - Research inspiration from:

    - Vaswani et al., “Attention is All You Need” (2017).

    - Relevant academic blogs and GitHub repositories.

This repository is for educational and research purposes only.
