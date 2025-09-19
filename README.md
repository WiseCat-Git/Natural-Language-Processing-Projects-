# Natural-Language-Processing-Projects

# Natural Language Processing Projects - Master's in Applied AI

This repository showcases advanced Natural Language Processing implementations developed during my **Master's in Applied Artificial Intelligence** at IEP + Summa University. These projects demonstrate progressive mastery from traditional NLP pipelines to cutting-edge transformer architectures, with emphasis on real-world applications and production-ready implementations.

## Academic Context & Learning Trajectory

### Unit 1: NLP Pipeline Fundamentals
**File:** `solucion_caso_practico_iep_iaa_nlp_u1.py`

**Core Foundations:**
- Complete NLP pipeline implementation (preprocessing → modeling → evaluation)
- Text preprocessing and normalization techniques
- Feature extraction and vectorization methods
- Traditional machine learning classification

### Unit 2: Sequential Deep Learning
**File:** `solucion_caso_práctico_iep_iaa_nlp_u2.py`

**Advanced Architectures:**
- Recurrent Neural Networks (LSTM/GRU) for sequence modeling
- Word embeddings and distributed representations
- Contextual sequence processing and sentiment analysis
- Deep learning optimization techniques

### Unit 3: Transformer Architectures
**File:** `solución_caso_práctico_iep_iaa_nlp_u3.py`

**State-of-the-Art Techniques:**
- Question-answering systems with BERT/DistilBERT
- Fine-tuning pre-trained transformer models
- Transfer learning and domain adaptation
- Advanced evaluation methodologies for complex NLP tasks

### Capstone Project: Production NLP System
**File:** `solución_proyecto_de_aplicacion_iep_iaa_nlp.py`

**Industry-Grade Implementation:**
- End-to-end sentiment analysis system
- Multi-model comparison and ensemble techniques
- Production deployment considerations
- Comprehensive evaluation framework

## Project Portfolio Deep Dive

### 1. Amazon Reviews Sentiment Analysis Pipeline
**File:** `solucion_caso_practico_iep_iaa_nlp_u1.py`

**Objective:** Build complete NLP pipeline for product review sentiment classification

#### Technical Implementation Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Text Preprocessing** | NLTK, Regex | Tokenization, stopword removal, lemmatization |
| **Feature Extraction** | TF-IDF Vectorizer | Document representation (5000 features) |
| **Classification** | Random Forest | Multi-class sentiment prediction |
| **Evaluation** | Scikit-learn metrics | Accuracy, precision, recall, F1-score |

#### Key Technical Achievements
- **Dataset Processing:** Amazon reviews with comprehensive text cleaning
- **Pipeline Architecture:** Modular preprocessing → vectorization → classification
- **Performance Metrics:** 79.8% accuracy with detailed confusion matrix analysis
- **Class Imbalance Handling:** Identified and analyzed dataset skew toward positive reviews

#### Research Insights & Critical Analysis
- **Challenge Identified:** Severe class imbalance (majority class 5.0 stars)
- **Solution Proposed:** Binary classification and rebalancing techniques
- **Production Considerations:** Recommended SMOTE, cross-validation, and hyperparameter tuning
- **Business Impact:** Automated review classification system ready for e-commerce integration

### 2. Advanced Sentiment Analysis with Sequential Models
**File:** `solucion_caso_práctico_iep_iaa_nlp_u2.py`

**Objective:** Compare representation methods and implement deep sequential models

#### Multi-Modal Representation Analysis
| Representation | Technology | Dimensionality | Context Awareness |
|----------------|------------|----------------|-------------------|
| **Bag of Words** | CountVectorizer | Sparse, high-dim | No |
| **TF-IDF** | TfidfVectorizer | Sparse, weighted | No |
| **Word2Vec** | Gensim | Dense, 100-dim | Local context |
| **Sentence Transformers** | HuggingFace | Dense, 768-dim | Global context |

#### Deep Learning Architecture
```python
# LSTM Implementation
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3-class sentiment
])
```

#### Performance Comparison Results
- **LSTM Accuracy:** 95% with excellent sequence modeling
- **GRU Performance:** Marginal improvement in training efficiency
- **Contextual Superiority:** Sentence Transformers showed superior semantic clustering
- **Visualization:** PCA/t-SNE plots demonstrated clear sentiment separation

### 3. Question-Answering with Transformer Fine-tuning
**File:** `solución_caso_práctico_iep_iaa_nlp_u3.py`

**Objective:** Implement production-grade QA system using BERT architecture

#### Model Evaluation & Selection
| Model | Architecture | Parameters | SQUAD Performance |
|-------|--------------|------------|-------------------|
| **DistilBERT-SQUAD** | Transformer-Encoder | 66M | Best baseline performance |
| **BERT-Large-WWM** | Transformer-Encoder | 340M | High accuracy, resource intensive |
| **RoBERTa-Base** | Optimized Transformer | 125M | Strong generalization |
| **ALBERT-Base-v2** | Factorized Embedding | 12M | Parameter efficiency |

#### Fine-tuning Implementation
```python
# Training Configuration
training_args = TrainingArguments(
    output_dir="./qa_model_finetuned",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)
```

#### Research Results & Impact
- **Training Efficiency:** 77 minutes for 2 epochs with significant loss reduction
- **Performance Improvement:** Training loss: 0.573 → 0.314 (46% reduction)
- **Confidence Scores:** High-quality predictions (0.88-0.99 confidence range)
- **Production Readiness:** Model saved and deployed via HuggingFace pipeline

### 4. Production Sentiment Classification System
**File:** `solución_proyecto_de_aplicacion_iep_iaa_nlp.py`

**Objective:** Build comprehensive, production-ready sentiment analysis system

#### Multi-Model Architecture Comparison
| Model Type | Implementation | Accuracy | F1-Score | Training Time |
|------------|----------------|----------|----------|---------------|
| **LSTM** | TensorFlow/Keras | 1.000 | 1.000 | 5 epochs |
| **GRU** | TensorFlow/Keras | 1.000 | 1.000 | 5 epochs |
| **DistilBERT** | HuggingFace Pipeline | 0.690 | 0.676 | Pre-trained |

#### Production Pipeline Features
```python
class SentimentAnalysisSystem:
    def __init__(self):
        self.preprocessor = SimplePreprocessor()
        self.w2v_model = Word2Vec(vector_size=100, window=5, min_count=2)
        self.ft_model = FastText(vector_size=100, window=5, min_count=2)
        self.lstm_classifier = self.build_lstm()
        self.bert_pipeline = pipeline("sentiment-analysis")
    
    def predict_sentiment(self, text):
        # Multi-model ensemble prediction
        processed_text = self.preprocessor.preprocess(text)
        lstm_pred = self.lstm_classifier.predict(processed_text)
        bert_pred = self.bert_pipeline(text)
        return self.ensemble_decision(lstm_pred, bert_pred)
```

#### Advanced Technical Features
- **Embedding Integration:** Word2Vec and FastText for semantic understanding
- **Multi-Model Ensemble:** LSTM/GRU + Transformer combination
- **Production Optimizations:** Batch processing, confidence scoring, error handling
- **Deployment Ready:** Modular architecture for API integration

## Technical Expertise Demonstrated

### Natural Language Processing Fundamentals
- **Text Preprocessing:** Comprehensive cleaning, tokenization, normalization pipelines
- **Feature Engineering:** TF-IDF, n-grams, POS tagging, dependency parsing
- **Traditional ML:** Naive Bayes, SVM, Random Forest for text classification
- **Evaluation:** Comprehensive metrics including confusion matrices and class-specific analysis

### Deep Learning for NLP
- **Word Embeddings:** Word2Vec, FastText, GloVe implementation and comparison
- **Sequential Models:** LSTM, GRU, bidirectional variants with dropout and regularization
- **Attention Mechanisms:** Understanding of self-attention in transformer architectures
- **Transfer Learning:** Fine-tuning pre-trained models for domain-specific tasks

### Transformer Architecture Mastery
- **Pre-trained Models:** BERT, DistilBERT, RoBERTa, ALBERT implementation
- **Fine-tuning Strategies:** Learning rate scheduling, layer freezing, gradient accumulation
- **HuggingFace Ecosystem:** Transformers library, datasets, tokenizers integration
- **Advanced Applications:** Question answering, sentiment analysis, text generation

### Production Engineering
- **Pipeline Design:** Modular, scalable architectures for real-world deployment
- **Performance Optimization:** Model compression, inference acceleration, batch processing
- **Error Handling:** Robust preprocessing, fallback mechanisms, confidence thresholding
- **Evaluation Frameworks:** Comprehensive testing, A/B testing setup, metric tracking

## Real-World Applications & Business Impact

### E-commerce & Customer Intelligence
- **Review Analysis:** Automated sentiment classification for product feedback
- **Customer Support:** Intent recognition and response generation
- **Recommendation Systems:** Sentiment-aware product suggestions
- **Market Research:** Brand sentiment tracking and competitive analysis

### Information Retrieval & Knowledge Systems
- **Question Answering:** Customer service automation and knowledge base querying
- **Document Classification:** Content categorization and information extraction
- **Search Enhancement:** Semantic search and relevance ranking
- **Content Moderation:** Automated filtering and toxicity detection

### Research & Development
- **Model Comparison:** Systematic evaluation of traditional vs. modern approaches
- **Transfer Learning:** Domain adaptation techniques for specialized applications
- **Ensemble Methods:** Multi-model combination for improved performance
- **Interpretability:** Understanding model decisions and bias detection

## Technology Stack & Implementation Details

### Core Libraries & Frameworks
```python
# NLP Processing
import nltk                    # Traditional NLP toolkit
import spacy                   # Industrial-strength NLP
import gensim                  # Topic modeling and embeddings
from transformers import *     # State-of-the-art transformers

# Deep Learning
import tensorflow as tf        # Neural network implementation
import torch                   # PyTorch for transformer models
from keras.models import *     # High-level neural network API

# Data Processing & Visualization
import pandas as pd           # Data manipulation
import numpy as np            # Numerical computing
import matplotlib.pyplot as plt # Visualization
import seaborn as sns         # Statistical visualization
```

### Advanced Implementation Patterns
- **Modular Architecture:** Separation of preprocessing, modeling, and evaluation
- **Configuration Management:** YAML/JSON configuration for hyperparameters
- **Logging & Monitoring:** Comprehensive tracking of model performance
- **Version Control:** Model versioning and experiment tracking

## Performance Benchmarks & Evaluation

### Dataset Statistics
| Project | Dataset Size | Classes | Vocabulary Size | Avg. Text Length |
|---------|--------------|---------|-----------------|------------------|
| **Amazon Reviews** | 3,000 reviews | 3 (pos/neg/neu) | 5,000 features | 120 words |
| **Sequential Models** | 3,000 reviews | 3 (pos/neg/neu) | 65 unique tokens | 6.6 tokens |
| **QA System** | 98K examples | N/A (span prediction) | 30K+ tokens | 120 words |
| **Production System** | 3,000 reviews | 3 (pos/neg/neu) | 5,000 features | Variable |

### Model Performance Summary
| Approach | Best Accuracy | F1-Score | Training Time | Inference Speed |
|----------|---------------|----------|---------------|-----------------|
| **Traditional ML** | 79.8% | Variable | Minutes | Milliseconds |
| **Sequential DL** | 95.0% | 95.0% | Minutes | Seconds |
| **Transformers** | 69.0% | 67.6% | Hours | Seconds |
| **Ensemble** | Variable | Variable | Combined | Optimized |

## Research Contributions & Academic Value

### Methodological Innovations
- **Systematic Comparison:** Comprehensive evaluation across multiple paradigms
- **Production Focus:** Real-world deployment considerations throughout development
- **Interpretability:** Analysis of model decisions and failure cases
- **Scalability:** Architecture designs for large-scale applications

### Educational Impact
- **Progressive Learning:** Clear progression from fundamentals to advanced techniques
- **Practical Implementation:** Working code demonstrating theoretical concepts
- **Critical Analysis:** Honest assessment of strengths and limitations
- **Industry Relevance:** Focus on practical applications and business value

## Future Research Directions

### Technical Enhancements
- **Large Language Models:** Integration of GPT-4, Claude, and other frontier models
- **Multimodal NLP:** Text + image/audio processing for richer understanding
- **Few-shot Learning:** Adaptation to new domains with minimal training data
- **Efficient Architectures:** Model compression and edge deployment optimization

### Application Expansion
- **Multilingual Systems:** Cross-lingual transfer and code-switching handling
- **Domain Specialization:** Medical, legal, financial text processing
- **Real-time Processing:** Streaming text analysis and live sentiment tracking
- **Ethical AI:** Bias detection, fairness metrics, and responsible deployment

## Professional Impact & Career Development

### Technical Leadership
- **Architecture Design:** Led end-to-end system design from requirements to deployment
- **Research & Development:** Conducted systematic comparison of multiple approaches
- **Knowledge Transfer:** Documented implementations for team knowledge sharing
- **Innovation:** Applied cutting-edge research to practical business problems

### Industry Readiness
- **Production Systems:** Built deployable models with proper error handling
- **Performance Optimization:** Achieved significant improvements through fine-tuning
- **Scalable Design:** Created architectures suitable for enterprise deployment
- **Best Practices:** Implemented comprehensive evaluation and monitoring frameworks

---

**Institution:** IEP + Summa University  
**Program:** Master's in Applied Artificial Intelligence  
**Academic Year:** 2024-2025  
**Specialization:** Natural Language Processing & Deep Learning

*This repository demonstrates mastery of NLP from fundamental text processing to state-of-the-art transformer implementations, with strong emphasis on practical applications and production deployment considerations.*
