# CLAF-Net: Cross-Lingual Asymmetric Focal Network for Zero-Shot Disaster Response Classification

> **A multilingual, zero-shot disaster-response classification framework designed to reduce translation latency and improve detection of rare, high-risk disaster events.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/docs/transformers/)
[![Model](https://img.shields.io/badge/Model-XLM--RoBERTa-orange)](https://huggingface.co/xlm-roberta-base)
[![Task](https://img.shields.io/badge/Task-Multilabel%20Classification-green)](#problem)
[![Domain](https://img.shields.io/badge/Domain-Disaster%20Response-red)](#overview)

## 📌 Overview

**CLAF-Net (Cross-Lingual Asymmetric Focal Network)** is a multilingual NLP framework for automated disaster-response message classification.

The system addresses two major challenges in real-time disaster triage:

1. **Translation latency** — traditional systems translate non-English messages into English before classification.
2. **Extreme class imbalance** — common disaster-related categories greatly outnumber rare but critical events such as `fire`, `missing_people`, `hospitals`, and `tools`.

CLAF-Net combines:

* **XLM-RoBERTa** for multilingual representation learning
* **Context Fusion augmentation** for improved cross-lingual transfer
* **Asymmetric Focal Loss** for handling long-tailed class distributions
* **Triage Confidence Routing** for human-in-the-loop handling of uncertain predictions

The paper reports a **Macro F1 of 0.68** and approximately **10× lower inference latency** compared with a translation-based BERT pipeline.

---

## 🚨 Problem

Disaster-response systems receive large volumes of multilingual messages from sources such as:

* Social media
* SMS gateways
* Digital reporting systems
* Crisis communication channels

Conventional approaches face two important limitations.

### 1. Translation Bottleneck

A traditional pipeline can be represented as:

```text
Native Language
      ↓
Machine Translation API
      ↓
English Text
      ↓
BERT Classifier
      ↓
Disaster Categories
```

This introduces additional latency and creates a dependency on external translation services.

CLAF-Net instead performs inference directly on the original language:

```text
Native Language
      ↓
XLM-RoBERTa
      ↓
Multi-Label Classifier
      ↓
Disaster Categories
```

### 2. Long-Tailed Class Imbalance

Some categories occur very frequently while critical categories appear in fewer than 1% of messages.

Examples include:

* `related`
* `weather_related`
* `food`

versus rare categories such as:

* `missing_people`
* `fire`
* `hospitals`
* `tools`

The paper reports approximately **30,000 messages across 36 disaster categories** with a strongly skewed class distribution.

---

# 🧠 CLAF-Net Architecture

```text
                  Input Message
                       │
                       ▼
              ┌─────────────────┐
              │ Context Fusion  │
              │   Augmentation  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   XLM-RoBERTa   │
              │    Encoder      │
              └────────┬────────┘
                       │
                       ▼
                 [CLS] Vector
                       │
                       ▼
                 Dropout (0.1)
                       │
                       ▼
              Fully Connected Layer
                       │
                       ▼
                    Sigmoid
                       │
                       ▼
             36 Disaster Categories
                       │
                       ▼
            Triage Confidence Routing
                 │             │
             Confident       Unsure
                 │             │
                 ▼             ▼
            Automatic      Human Review
             Routing
```

The architecture uses `xlm-roberta-base`, extracts the final `[CLS]` representation, applies dropout, and projects the representation into the 36-category label space.

---

# 🌍 Context Fusion

CLAF-Net introduces a **Context Fusion** strategy during training.

A native-language message and its English translation are concatenated:

```text
[CLS]
Native Text
[SEP]
English Translation
[SEP]
```

This encourages the model to learn semantic alignment between native-language disaster terminology and its English representation.

During actual inference, however, the translation API is **not required**:

```text
[CLS]
Native Text
[SEP]
```

This allows direct multilingual inference.

---

# 🎯 Asymmetric Focal Loss

Standard Binary Cross-Entropy can become dominated by large numbers of easy negative examples in highly imbalanced datasets.

CLAF-Net replaces BCE with an asymmetric Focal Loss:

$$
FL(p_t)=-\alpha_t(1-p_t)^\gamma\log(p_t)
$$

Parameters used in the experiments:

```text
γ = 2.0
α = 0.25
```

The focusing parameter reduces the contribution of easy examples, allowing the model to focus more strongly on difficult and minority-class examples.

---

# 🛡️ Triage Confidence Routing

Because disaster-response classification can involve high-consequence errors, CLAF-Net does not force every prediction into a binary decision.

Each class has a validation-derived threshold:

```text
τc
```

Predictions close to the threshold are marked as:

```text
UNSURE
```

and routed to human operators.

```text
             Model Prediction
                    │
                    ▼
          Compare with τc
             /          \
            /            \
     Confident          Unsure
         │                 │
         ▼                 ▼
  Automated Route     Human Review
```

This creates a **human-in-the-loop** deployment strategy for ambiguous cases.

---

# 📊 Dataset

The research uses a multilingual, multi-label disaster-response dataset containing approximately:

| Property   |                                  Value |
| ---------- | -------------------------------------: |
| Messages   |                                ~30,000 |
| Categories |                                     36 |
| Training   |                                    70% |
| Validation |                                    10% |
| Testing    |                                    20% |
| Languages  | English, Haitian Creole, Spanish, Urdu |
| Task       |             Multi-label classification |

The dataset includes messages originating from crises such as the **2010 Haiti Earthquake, Pakistan floods, and hurricanes in the Americas**.

---

# 🔬 Experimental Setup

### Baselines

CLAF-Net was compared against:

1. **Pipeline-BERT (BCE)**
2. **mBERT (BCE)**
3. **XLM-R (BCE)**

The XLM-R baseline uses the same `xlm-roberta-base` architecture but without Context Fusion and Asymmetric Focal Loss.

### Training Configuration

| Parameter         | Configuration                 |
| ----------------- | ----------------------------- |
| Framework         | PyTorch 1.13                  |
| Transformers      | HuggingFace Transformers 4.25 |
| GPU               | NVIDIA A100 40GB              |
| Optimizer         | AdamW                         |
| Learning Rate     | `2 × 10⁻⁵`                    |
| Batch Size        | 32                            |
| Epochs            | 5                             |
| Warmup            | 10%                           |
| Gradient Clipping | 1.0                           |
| Early Stopping    | Patience = 2                  |
| Focal γ           | 2.0                           |
| Focal α           | 0.25                          |
| Runs              | 3                             |

---

# 📈 Results

## Overall Performance

| Model               | Micro F1 | Macro F1 | Latency / Message |
| ------------------- | -------: | -------: | ----------------: |
| Pipeline-BERT (BCE) |     0.68 |     0.41 |            450 ms |
| mBERT (BCE)         |     0.65 |     0.38 |             40 ms |
| XLM-R (BCE)         |     0.69 |     0.42 |             45 ms |
| **CLAF-Net**        | **0.72** | **0.68** |         **45 ms** |

The paper reports that CLAF-Net improves Macro F1 from **0.42 to 0.68** over the XLM-R BCE baseline and reduces latency from **450 ms to 45 ms** compared with the translation-based Pipeline-BERT system.

---

# 🧪 Ablation Study

The contribution of each major component was evaluated by removing it from the architecture.

| Configuration                    | Macro F1 |
| -------------------------------- | -------: |
| **CLAF-Net — Full Architecture** | **0.68** |
| − Context Fusion                 |     0.60 |
| − Asymmetric Focal Loss / BCE    |     0.44 |
| − Both / Base XLM-R              |     0.42 |

The reported results show that removing Context Fusion decreases Macro F1 from `0.68 → 0.60`, while replacing Asymmetric Focal Loss with BCE decreases it to `0.44`.

---

# 🔥 Minority-Class Detection

One of the primary objectives of CLAF-Net is improving detection of rare disaster categories.

Reported Average Precision values include:

| Category          | Average Precision |
| ----------------- | ----------------: |
| `weather_related` |              0.88 |
| `water`           |              0.40 |
| `fire`            |              0.03 |
| `missing_people`  |              0.02 |

The paper notes that the BCE baselines produced an AP of `0.00` for `fire` and `missing_people`, while CLAF-Net established a functioning detection baseline for these rare categories.

---

# ⚠️ Failure Cases

CLAF-Net still has identifiable failure modes.

### False Positives

The model can confuse:

* Requests for medical assistance
* General reports about medical situations

For example, reports about doctors treating patients may be classified as `medical_help` even when the message is only providing situational information.

### False Negatives

Errors can occur when messages contain:

* Heavy sarcasm
* Local slang
* Highly corrupted SMS text
* Metaphorical language

These limitations motivate the human-review mechanism for uncertain predictions.

---

# ⚙️ Installation

```bash
git clone https://github.com/<your-username>/CLAF-Net.git

cd CLAF-Net

python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Suggested Project Structure

```text
CLAF-Net/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── models/
│   ├── claf_net.py
│   ├── focal_loss.py
│   └── classifier.py
│
├── preprocessing/
│   ├── context_fusion.py
│   └── tokenizer.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── config.py
│
├── inference/
│   ├── predict.py
│   └── triage_router.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── results/
│   ├── metrics/
│   └── figures/
│
├── requirements.txt
├── README.md
└── LICENSE
```

> **Note:** This structure is a suggested implementation layout based on the architecture described in the paper; the paper itself does not specify this exact repository structure.

---

# 🚀 Inference Workflow

A typical inference pipeline is:

```python
message
   ↓
XLM-R Tokenizer
   ↓
XLM-RoBERTa
   ↓
CLS Representation
   ↓
Classification Head
   ↓
Sigmoid Probabilities
   ↓
Class-Specific Thresholds
   ↓
┌───────────────┬───────────────┐
│   Confident   │     Unsure    │
│       ↓       │       ↓       │
│ Automated     │ Human Review  │
│ Routing       │ Queue         │
└───────────────┴───────────────┘
```

---

# 💡 Example

### Input

```text
We are trapped inside the building. Please send help.
```

### Possible Output

```json
{
  "request": 0.94,
  "buildings": 0.81,
  "search_and_rescue": 0.89,
  "medical_help": 0.12
}
```

The exact output depends on the trained model, thresholds, and input message.

---

# 🔭 Future Work

The paper identifies several directions for future development:

* **Parameter-efficient fine-tuning (PEFT)**
* **LoRA-based adaptation**
* Rapid adaptation to new disaster scenarios
* **Multimodal disaster-response classification**
* Integration of image attachments
* Improved support for ultra-low-resource languages and regional dialects

---

# 🌐 Potential Applications

CLAF-Net can be explored for applications such as:

* 🚨 Emergency message triage
* 🌊 Flood response
* 🔥 Fire incident detection
* 🏥 Medical assistance identification
* 🔎 Missing-person reports
* 🚑 Search-and-rescue coordination
* 📱 Multilingual crisis communication
* 🛰️ Large-scale disaster information monitoring

---

# 👥 Authors

**Farah Hamna M**
Department of Information Technology
Karpagam College of Engineering, Coimbatore, India

**Rishikesh V**
Department of Information Technology
Karpagam College of Engineering, Coimbatore, India

**Shrutin K**
Department of Information Technology
Karpagam College of Engineering, Coimbatore, India

---

# 📚 References

The research paper references work covering:

* Focal Loss
* XLM-RoBERTa
* BERT
* SMOTE
* Crisis informatics
* Disaster-response NLP
* Transformer architectures
* Multilingual crisis-response systems

See the full research paper for the complete reference list.

---

# 📄 Research Paper

**CLAF-Net: A Cross-Lingual Asymmetric Focal Network for Zero-Shot Disaster Response Classification**

The complete research paper is included in this repository.

---

## ⭐ Key Takeaways

```text
                 CLAF-Net
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   XLM-RoBERTa  Context Fusion  Focal Loss
       │            │            │
       └────────────┼────────────┘
                    ▼
       Zero-Shot Multilingual
          Disaster Triage
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Lower Latency       Better Minority
                       Class Detection
```

**CLAF-Net combines multilingual zero-shot learning with asymmetric focal optimization to build a disaster-response classification system capable of directly processing native-language messages while addressing severe class imbalance.**
