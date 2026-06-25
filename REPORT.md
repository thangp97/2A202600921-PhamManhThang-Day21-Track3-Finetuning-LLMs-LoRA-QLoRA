# Lab 21 — Evaluation Report

**Học viên**: Phạm Mạnh Thắng — 2A202600921
**Ngày nộp**: 2026-06-25
**Submission option**: B (GitHub + HuggingFace Hub) ⭐ +5 bonus
**HF Hub adapter (r=16)**: https://huggingface.co/hondadream/qwen2.5-3b-vi-lab21-r16

---

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Qwen2.5-3B, load 4-bit NF4 — QLoRA)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 mẫu (180 train + 20 eval), split 90/10 seed=42
- **Format**: Alpaca (`### Instruction / ### Input / ### Response`), auto-detect cột `_vi` suffix
- **max_seq_length**: 1024 (token length: p50=227, **p95=562**, p99=704 → cap 1024)
- **GPU**: Tesla T4, 15 GB VRAM (Google Colab Free)
- **Training config**: 3 epochs, cosine LR = 2e-4, warmup ratio 0.10, effective batch size 8 (1 × grad_accum 8), optimizer `adamw_8bit`, gradient checkpointing (unsloth)
- **LoRA target**: `["q_proj", "v_proj"]` (lab spec), dropout 0
- **Training cost**: ~$0.06 (10.9 phút tổng cho 3 ranks @ $0.35/hr)

---

## 2. Rank Experiment Results

| Rank | alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|------------------|------------|-----------|-----------|------------|
| 8    | 16    | 1,843,200        | 3.65 min   | 9.32 GB   | 1.5577    | 4.748      |
| 16   | 32    | 3,686,400        | 3.59 min   | 8.73 GB   | 1.5161    | 4.554      |
| 64   | 128   | 14,745,600       | 3.62 min   | 10.10 GB  | 1.4768    | **4.379**  |
| Base | -     | -                | -          | -         | N/A*      | N/A*       |

\* *Base model perplexity không đo được do giới hạn VRAM/thời gian của session T4; so sánh định tính base vs fine-tuned ở Section 4.*

**Nhận xét nhanh**: tăng rank đơn điệu làm **giảm perplexity** (4.748 → 4.554 → 4.379) và **tăng trainable params tuyến tính** (×2 mỗi bước rank, r=64 nhiều gấp 8 lần r=8). Train time gần như không đổi (~4 min) vì với dataset nhỏ (180 mẫu, 69 steps) chi phí bị chi phối bởi forward/backward của base model bị đóng băng, không phải bởi số tham số adapter.

---

## 3. Loss Curve Analysis

[Đính kèm `results/loss_curve.png`]

- **Quan sát**: Training loss giảm đều từ ~1.6 → ~1.4 qua 69 steps (3 epochs). Đường cong mượt, giảm đơn điệu, không dao động bất thường.
- **Overfitting?** **Không có dấu hiệu rõ ràng**. Trên T4, eval-during-training bị tắt để tiết kiệm VRAM nên chỉ có train loss curve; tuy nhiên eval loss cuối (1.48–1.56) vẫn nằm sát train loss cuối (~1.4) → khoảng cách train/eval nhỏ, model **chưa học thuộc lòng**. Với dataset chỉ 180 mẫu và LoRA (chỉ ~0.12% params), nguy cơ overfit thấp; nếu train thêm nhiều epoch hơn (>5) thì mới cần theo dõi eval loss bật lên.
- **Vì sao loss bắt đầu ở ~1.6 chứ không < 1**: đây là cross-entropy loss trên dự đoán token tiếp theo (perplexity = e^loss ≈ 5). Ngôn ngữ tự nhiên vốn khó đoán; loss 1.5–2.5 là vùng lành mạnh cho LLM, loss < 1 thường là dấu hiệu overfit hoặc dataset quá lặp.

---

## 4. Qualitative Comparison (5 examples)

So sánh **base Qwen2.5-3B** vs **fine-tuned r=16** (cùng prompt, max_new_tokens=200).

### Example 1 — Giải thích machine learning cho người mới
- **Base**: "Machine learning là một phân khúc của trí tuệ nhân tạo... tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu..."
- **Fine-tuned**: "Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng..."
- **Nhận xét**: ~ Tương đương. FT diễn đạt mạch lạc, tự nhiên hơn một chút ("không có sự hướng dẫn trực tiếp" — đúng tinh thần unsupervised), nhưng cả hai đều chính xác. **Improved nhẹ về văn phong.**

### Example 2 — Code Python Fibonacci thứ n
- **Base**: Đưa hàm đệ quy, có check `if n <= 0: return "N p..."` (xử lý lỗi hơi cẩu thả).
- **Fine-tuned**: Hàm có `raise ValueError("Input phải là một số nguyên dương.")` + nhánh `elif n == 0` rõ ràng.
- **Nhận xét**: **Improved** — FT xử lý edge case bằng exception đúng chuẩn Python (raise thay vì trả string), code production hơn.

### Example 3 — Liệt kê 5 nguyên tắc thiết kế UI/UX
- **Base**: "1. Thân thiện với người dùng... 2..." — văn phong lặp từ "thân thiện" nhiều lần.
- **Fine-tuned**: "1. Chuyển đổi... 2. Thích ứng... 3. Đơn giản..." — danh sách ngắn gọn, mỗi nguyên tắc một từ khóa.
- **Nhận xét**: **Improved về format** — FT bám sát yêu cầu "liệt kê", trình bày dạng bullet súc tích hơn, ít lặp từ.

### Example 4 — Tóm tắt khác biệt LoRA vs QLoRA
- **Base**: "LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA)... sử dụng các phép biến đổi thấp độ phức tạp." — định nghĩa **đúng**.
- **Fine-tuned**: "LoRA (Layer-wise Adaptive Regularization Optimization)..." — **sai tên viết tắt** (LoRA = Low-Rank Adaptation, không phải "Layer-wise Adaptive Regularization").
- **Nhận xét**: **Degraded** — đây là một **case loss**: fine-tune trên dataset general Việt khiến model bịa lại định nghĩa thuật ngữ kỹ thuật. Cho thấy fine-tune **không thêm knowledge**, đôi khi làm nhiễu kiến thức sẵn có (đúng "quy tắc vàng": fine-tune cho style, RAG cho knowledge).

### Example 5 — Phân biệt prompt engineering, RAG, fine-tuning
- **Base**: "...ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện..." (hơi lặp "cải thiện hiệu suất").
- **Fine-tuned**: "...ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt)..."
- **Nhận xét**: **Improved** — FT phân tách 3 khái niệm rõ ràng hơn, định nghĩa prompt engineering chính xác hơn.

**Tổng kết qualitative**: 3 improved (style/format/code), 1 tương đương, 1 degraded (bịa thuật ngữ). Fine-tune cải thiện **văn phong và format tiếng Việt** nhưng **không bổ sung kiến thức** — và có rủi ro làm nhiễu fact kỹ thuật.

---

## 5. Conclusion về Rank Trade-off

Trên dataset Vietnamese-Alpaca 200 mẫu này, **r=16 cho ROI tốt nhất**. Xét cụ thể: r=8 → r=16 giảm perplexity 4.748 → 4.554 (~4%) với chi phí gấp đôi params; r=16 → r=64 chỉ giảm tiếp 4.554 → 4.379 (~3.8%) nhưng tốn gấp **4 lần** trainable params (3.7M → 14.7M) và **+1.4 GB peak VRAM** (8.73 → 10.10 GB). Đây là **diminishing returns** rõ rệt: lợi ích perplexity cận biên co lại nhanh trong khi chi phí bộ nhớ và rủi ro overfit (capacity dư thừa trên dataset nhỏ) tăng tuyến tính. Điểm "gãy" của đường cong nằm quanh r=16 — vượt qua đó, mỗi tham số adapter thêm vào mang lại ngày càng ít cải thiện vì rank thực sự cần để biểu diễn ΔW cho task style-transfer này thấp hơn nhiều so với 64.

**Nếu deploy production**: tôi chọn **r=16**. Nó nằm ở sweet spot — perplexity gần bằng r=64 (4.55 vs 4.38, chênh < 4%) nhưng nhẹ hơn nhiều về VRAM và adapter size (3.7M params), nạp/serve nhanh hơn, ít rủi ro overfit. r=64 chỉ đáng cân nhắc khi dataset lớn hơn (vài nghìn mẫu chất lượng cao) và task phức tạp hơn (đa kỹ năng) để capacity dư thừa được tận dụng. r=8 phù hợp khi VRAM cực kỳ eo hẹp hoặc cần nhiều adapter multi-tenant trên một GPU.

---

## 6. What I Learned

- **Rank không phải "càng cao càng tốt"**: tăng rank 8×(8→64) chỉ giảm perplexity ~8% — minh chứng thực nghiệm cho ý tưởng cốt lõi của LoRA là cập nhật trọng số thực sự nằm trong không gian low-rank, nên rank vừa phải đã đủ.
- **Fine-tune cải thiện style/format, KHÔNG thêm knowledge**: Example 4 (model bịa lại định nghĩa LoRA sau khi fine-tune) cho tôi thấy trực tiếp tại sao "RAG cho knowledge, fine-tune cho style" — và tại sao phải đánh giá cả case loss, không chỉ cherry-pick case win.
- **Perplexity ≈ e^loss và loss ~1.5 là bình thường**: trước đây tôi tưởng loss phải < 1; giờ hiểu loss là cross-entropy trên token tiếp theo, và con số tuyệt đối ít quan trọng bằng xu hướng giảm + so sánh tương đối giữa các cấu hình.

---

> Số liệu trong report được trích trực tiếp từ output notebook `Lab21_LoRA_Finetuning_T4.ipynb` (run trên Colab T4). File `results/rank_experiment_summary.csv` và `results/qualitative_comparison.csv` chứa số liệu đầy đủ để verify.
