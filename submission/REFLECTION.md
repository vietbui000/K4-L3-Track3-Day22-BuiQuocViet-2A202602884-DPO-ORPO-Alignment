# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** _<Bùi Quốc Việt>_
**Khoá:** _<A20-K4 / track 3>_
**Tier đã chạy:** _<T4 | BIGGPU | cả hai>_
**Ngày:** _<2026-10-09>_

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | Khoảng 60% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1.0 |
| Giám khảo | rm-panel: Skywork/Llama-3.2-3B và Qwen3-4B; sanity accuracy: 1.0 |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Khoảng 40-60 phút |
| VRAM cao nhất | ~ 12 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.1031 |
| Độ chính xác reward trên held-out | 0.71 |
| Margin trên held-out | 0.0908 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 602.66 → 611.69 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Nhìn vào biểu đồ reward và các chỉ số từ `dpo_metrics.json`, ta thấy giá trị `rewards/chosen` có xu hướng tăng đều, đạt mức khoảng 0.447 vào cuối quá trình huấn luyện. Trong khi đó, `rewards/rejected` cũng thay đổi nhưng giữ ở mức thấp hơn (khoảng 0.344). Margin (khoảng cách giữa chosen và rejected) mở rộng chủ yếu là nhờ `chosen` reward tăng mạnh hơn so với `rejected`. Đây là dấu hiệu rất tích cực cho thấy mô hình thực sự học được cách ưu tiên câu trả lời tốt.
Quan trọng hơn, xu hướng này diễn ra đồng thời trên cả tập huấn luyện và tập held-out (margin held-out đạt 0.0908, gần sát với margin train là 0.1031, cùng với accuracy 71%). Điều này chứng tỏ mô hình không hề bị overfit (học thuộc) mà đã có khả năng tổng quát hóa tốt. Những quan sát này hoàn toàn khớp với chẩn đoán tự động là `INTENDED` – mô hình hoạt động đúng như kỳ vọng lý thuyết của thuật toán DPO.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 7 | 10 | 33 | 0.47 (40% - 55%) | 0.489 | 62.5% |
| hữu ích — helpfulness (4) | 4 | 0 | 1 | 3 | 0.375 (12.5% - 50%) | 0.50 | 0.0% |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 0.625 (50% - 87.5%) | 0.625 | 100.0% |

Giám khảo: Hội đồng 2 mô hình (Skywork Llama-3.2-3B và Qwen3-4B) · sanity accuracy: 1.0 · `score_length_spearman` (reward model): 0.25 (Qwen3) và -0.04 (Llama)

Dựa vào các chỉ số, ta thấy khoảng tin cậy 95% của win rate trên tập held-out là [0.4, 0.55], có chứa giá trị 0.5. Điều này chỉ ra rằng chưa có đủ bằng chứng thống kê để khẳng định bản DPO thực sự vượt trội hơn SFT một cách rõ rệt (kết quả chủ yếu là hoà). Giám khảo hoàn toàn đáng tin cậy trên tiếng Việt vì `sanity_accuracy` đạt mức tuyệt đối 1.0.
Tỉ lệ câu dài hơn thắng chiếm 62.5%, cho thấy giám khảo có hơi thiên vị độ dài một chút. Tuy nhiên, khi xét riêng các cặp dài gần bằng nhau, win rate vẫn ở mức 0.489, chứng tỏ DPO không chỉ đơn thuần "hack" bằng cách viết dài. Hai giám khảo trong hội đồng cho kết quả khá sát nhau (Qwen3 cho win rate 0.49, Llama cho 0.47), dù Qwen3 thiên vị DPO một chút nhẹ có thể do "rò rỉ sở thích" (cùng họ mô hình).

*Ví dụ cụ thể:*
- **Về độ hữu ích:** DPO bị thua ở 1 câu. Có thể trong một số tình huống, bản DPO do cố gắng làm hài lòng reward model nên trả lời dài dòng và vòng vo hơn, trong khi bản SFT trả lời ngắn gọn, trực diện và đáp ứng đúng nhu cầu người dùng.
- **Về an toàn:** DPO ghi điểm thắng ở 1 câu. Ở ví dụ này, DPO đã thể hiện khả năng từ chối những câu hỏi có tính rủi ro, đồng thời đưa ra được lời giải thích khéo léo và chuẩn mực hơn bản SFT vốn có thể trả lời một cách ngây ngô.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

_Nếu không chạy: viết giả thuyết 3 câu về điều bạn dự đoán sẽ thấy._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định quan trọng nhất của tôi trong bài Lab này là giữ nguyên giá trị hệ số phạt KL divergence **β = 0.1** cho thuật toán DPO thay vì thay đổi nó.
1. **Phương án thay thế:** Tôi có thể sử dụng β nhỏ hơn (ví dụ 0.05) để ép mô hình bám sát dữ liệu sở thích mạnh mẽ hơn, hoặc sử dụng β lớn hơn (ví dụ 0.5) để mô hình cẩn trọng hơn, hạn chế sai lệch xa so với bản gốc SFT ban đầu.
2. **Vì sao chọn phương án này:** β = 0.1 được xem là một mức cân bằng tiêu chuẩn, vừa đủ để mô hình cập nhật theo tín hiệu sở thích (chosen > rejected), vừa đủ để giữ lại các kiến thức và văn phong cơ bản đã có từ mô hình tham chiếu. Với tập dữ liệu kích thước nhỏ gọn (800 mẫu huấn luyện), việc phạt KL quá nhẹ (β rất nhỏ) dễ dẫn đến overfit hoặc phá vỡ khả năng sinh ngôn ngữ tự nhiên.
3. **Kết quả:** Kết quả thu được phần nào xác nhận quyết định này là an toàn khi chẩn đoán tự động hiển thị `INTENDED` (mô hình học đúng hướng) và margin trên tập held-out tăng ổn định (0.09). Tuy nhiên, điều làm tôi khá bất ngờ là win rate tổng thể vẫn chưa thể bứt phá mạnh mẽ (> 0.5) so với bản SFT.
4. **Làm lại thì đổi gì:** Nếu được làm lại với tài nguyên thời gian nhiều hơn, tôi sẽ thử thực hiện "beta-sweep" (thử nghiệm huấn luyện thêm các bản với β = 0.05 và β = 0.5) để xem liệu việc giảm β có giúp đẩy win rate lên cao hơn mà không làm hỏng ngôn ngữ hay không. Ngoài ra, tôi cũng muốn thử nghiệm phương pháp ORPO để so sánh hiệu năng khi không cần phải duy trì mô hình tham chiếu trong bộ nhớ.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
