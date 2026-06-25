# Lab 21 — Links (Submission Option B)

**Học viên**: Phạm Mạnh Thắng — 2A202600921  <!-- TODO: xác nhận MSSV/Họ tên -->

## 🤗 HuggingFace Hub — LoRA Adapter (r=16)

- **Adapter r=16**: https://huggingface.co/hondadream/qwen2.5-3b-vi-lab21-r16
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Status**: Public ✅

### Cách load adapter
```python
from peft import PeftModel
from unsloth import FastLanguageModel

base, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-3B-bnb-4bit", max_seq_length=1024, load_in_4bit=True)
model = PeftModel.from_pretrained(base, "hondadream/qwen2.5-3b-vi-lab21-r16")
```

## 💻 GitHub Repository

- **Repo**: https://github.com/thangp97/2A202600921-PhamManhThang-Day21-Track3-Finetuning-LLMs-LoRA-QLoRA

## 📁 Artifacts trong submission

| File | Mô tả |
|------|-------|
| `REPORT.md` | Evaluation report (6 sections) |
| `notebook.ipynb` | Notebook đã clear outputs |
| `results/rank_experiment_summary.csv` | Metrics 3 ranks (verify được) |
| `results/qualitative_comparison.csv` | 5 ví dụ base vs fine-tuned |
| `results/loss_curve.png` | Loss curve r=16 |
