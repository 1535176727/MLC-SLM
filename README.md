<div align="center">
    <h1>
  MLC-SLM
    </h1>
    <p>
    Official PyTorch code for paper <br>
    <b><em>BRIDGING THE GAP: A COMPARATIVE EXPLORATION OF SPEECH-LLM AND END-TO-END ARCHITECTURE FOR MULTILINGUAL CONVERSATIONAL ASR</em></b>
    </p>
    <p>
    </p>
    <a href="#"><img src="https://img.shields.io/badge/Paper-ArXiv-red" alt="paper"></a>
    <a href="https://huggingface.co/YuCeong-May/MLC-SLM/"><img src="https://img.shields.io/badge/Hugging%20Face-Model%20Page-yellow" alt="HF-model"></a>
    <a href="https://github.com/FireRedTeam/FireRedTTS"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache-2.0"></a>
</div>

# Dataset

The Task 1 dataset from the [MLC-SLM Challenge](https://www.nexdata.ai/competition/mlc-slm) contains about **1500 hours** of multilingual conversational speech across 15 categories (11 languages, with English further divided into 5 accents).

|       Category       | Hours (approx.) |
| :----------------------------: | :----------------------: |
|  English (American)  |       100       |
|  English (British)   |       100       |
| English (Australian) |       100       |
|   English (Indian)   |       100       |
| English (Philippine) |       100       |
|        French        |       100       |
|        German        |       100       |
|       Italian        |       100       |
|      Portuguese      |       100       |
|       Spanish        |       100       |
|       Japanese       |       100       |
|        Korean        |       100       |
|       Russian        |       100       |
|         Thai         |       100       |
|      Vietnamese      |       100       |
|      **Total**       |    **~1500**    |

**Data splits used in our experiments:**
- **Train**: the official ~1500h training set. In our setup, each category (100h) is divided into **98h for training** and **2h for validation**, resulting in a total of ~1470h train and ~30h valid.  
- **Valid**: the held-out **2h × 15 categories (≈30h)** subset from the training data, used for supervised model training and model selection.  
- **Dev**: the official development set, used only for final performance evaluation.  
- **Eval**: the [official evaluation set](https://huggingface.co/datasets/bsmu/MLC-SLM-Eval), used only for final performance evaluation.  
- **OOD**: sample 2 hours of test speech per language from the CommonVoice 21.0 dataset.   

# Architecture

![model](model.png)


# Training & Evaluation

The entire training and evaluation process is managed by the `run.sh` script, which organizes the pipeline into the following stages:  

- **Stage 0–3**: Data preparation  
- **Stage 4–5**: Projector training  
- **Stage 6–7**: ~~Joint training of projector and encoder~~ *(skipped in this project)*  
- **Stage 8–9**: Joint training of projector and LLM  
- **Stage 10**: Model averaging  
- **Stage 11**: Inference  
- **Stage 12**: Scoring

# Results

The main results of our proposed model compared with baselines and competition systems are shown below:

| **System**                | **Dev** | **Eval** | **CV-Test** |
|----------------------------|---------|----------|-------------|
| Whisper (LoRA-fine-tuned)  | 11.40   | 10.71    | **11.47**   |
| Whisper (Full-fine-tuned)  | 10.99   | 10.07    | 13.11       |
| mHuBERT (CTC)              | 29.67   | 19.99    | 68.49       |
| NTU-Speechlab              | 11.57   | 10.58    | -           |
| Seewoo                     | 12.73   | 11.57    | -           |
| SHNU-mASR                  | 13.39   | 11.43    | 19.86       |
| **Proposed Speech-LLM**    | 11.74   | **10.69**| 15.26       |

