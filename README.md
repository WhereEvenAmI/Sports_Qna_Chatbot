# Sports QA Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions about Cricket, Football, Basketball, and Badminton. Built with a FAISS semantic search pipeline and Microsoft Phi-2 (2.7B) fine-tuned using QLoRA.

---

## Architecture

```
User Query → Sport Detection → FAISS Retrieval → Phi-2 (LoRA) → Answer
```

1. **Sport Detection** — keyword-based classifier routes the query to the correct sport before retrieval, preventing cross-sport context pollution
2. **Semantic Retrieval** — MiniLM-L6-v2 encodes the query and searches a FAISS index of 30,000 sports QA embeddings for the most relevant contexts
3. **Answer Generation** — Phi-2 (2.7B), fine-tuned with QLoRA on 8,000 sports QA pairs, generates an answer grounded in the retrieved context

---

## Dataset

- **Source**: 4 Kaggle sports datasets (Cricket, Football, Basketball, Badminton)
- **Raw rows**: 64,685
- **After filtering** (empty answers removed): 41,030
- **Used for retrieval**: 30,000 (random sample)
- **Used for fine-tuning**: 8,000

---

## Fine-tuning

Base model: `microsoft/phi-2` (2.7B parameters)

| Config | Value |
|---|---|
| Method | QLoRA (4-bit NF4 quantization) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Trainable params | 5.2M (0.19% of total) |
| Training samples | 8,000 |
| Epochs | 1 |
| Batch size | 16 |
| Optimizer | paged_adamw_8bit |

Fine-tuned adapter published to HuggingFace Hub: [`NocturnoCulto67/phi2-sports-qa`](https://huggingface.co/NocturnoCulto67/phi2-sports-qa)

---

## Tech Stack

- Python, PyTorch
- `sentence-transformers` (MiniLM-L6-v2) — query encoding
- `faiss-cpu` — vector similarity search
- `transformers`, `peft`, `trl` — model loading, LoRA, fine-tuning
- `bitsandbytes` — 4-bit quantization
- `gradio` — chat interface
- Kaggle GPU (Tesla T4) — training environment

---

## Limitations

Accuracy is bounded by dataset coverage — questions about events not represented in the corpus may return incorrect or uncertain answers. The chatbot returns a fallback message when retrieval confidence is low rather than hallucinating.

---

## How to Run

The fine-tuned adapter is hosted on HuggingFace. To run locally:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

MODEL_ID   = "microsoft/phi-2"
ADAPTER_ID = "NocturnoCulto67/phi2-sports-qa"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer  = AutoTokenizer.from_pretrained(ADAPTER_ID)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto",
                                                   quantization_config=bnb_config,
                                                   trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
```

Then run all cells in `sports-qna-chatbot.ipynb` on Kaggle (GPU recommended).
