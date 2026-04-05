# Cheat Sheet Phần Image

File này là bản tóm tắt cực ngắn để học thuộc trước khi thuyết trình phần image.

---

## 1. Một câu mở đầu cho phần image

> Trong phần image, nhóm em giải bài toán phân loại giống chó trên dataset Stanford Dogs và so sánh công bằng hai họ mô hình là CNN với đại diện ResNet-50, và Vision Transformer với đại diện ViT-B/16.

---

## 2. Nói rất ngắn về bài toán

- Bài toán: `fine-grained image classification`
- Dataset: `Stanford Dogs`
- Số lớp: `120`
- Tổng số ảnh: `20,580`
- Mục tiêu: so sánh `ResNet-50` và `ViT-B/16` trên cùng protocol

Câu nói:

> Đây là bài toán fine-grained classification, tức là không chỉ phân biệt chó với mèo mà phải phân biệt giữa 120 giống chó khá giống nhau, nên model cần học được các chi tiết nhỏ như tai, mõm, bộ lông và hình dáng tổng thể.

---

## 3. Các con số phải nhớ

### Dataset

- Total: `20,580`
- Classes: `120`
- Official train: `12,000`
- Official test: `8,580`

### Split cuối cùng

- Train: `10,200`
- Val: `1,800`
- Test: `8,580`

### Loader lengths

- Train: `319`
- Val: `57`
- Test: `269`

### Input

- `224 x 224`
- `batch_size = 32`

### 2 model

- `ResNet-50`: `23.75M` params
- `ViT-B/16`: `85.89M` params

---

## 4. Một câu rất ngắn để giải thích CNN và Transformer

### CNN / ResNet-50

> ResNet-50 là CNN, nên nó nhìn ảnh theo kiểu cục bộ bằng các convolution filter, học dần từ cạnh, texture, đến bộ phận và cuối cùng là đặc trưng tổng thể của giống chó.

### Transformer / ViT-B/16

> ViT-B/16 là Vision Transformer, nên nó chia ảnh thành các patch 16x16 rồi dùng attention để mô hình hóa quan hệ giữa các patch trong toàn ảnh.

### Câu so sánh gọn nhất

> CNN mạnh về đặc trưng cục bộ, còn ViT mạnh về quan hệ toàn cục giữa các vùng ảnh.

---

## 5. Vì sao chọn ResNet-50 và ViT-B/16?

- đúng yêu cầu assignment: `CNN vs Vision Transformer`
- đều là backbone mạnh
- đều có pretrained ImageNet
- đều phù hợp cho transfer learning
- so sánh đủ rõ giữa 2 họ mô hình

Câu nói:

> Nhóm em chọn ResNet-50 và ViT-B/16 vì đây là hai backbone rất điển hình, đủ mạnh, có pretrained ImageNet, và đại diện rõ cho hai hướng CNN và Transformer trong computer vision.

---

## 6. EDA cần nói gì?

- dataset có độ biến thiên về kích thước ảnh, aspect ratio, độ sáng, màu sắc
- notebook khôi phục official split bằng `train_list.mat` và `test_list.mat`
- EDA giúp quyết định preprocessing và augmentation

Câu nói:

> Ở bước EDA, nhóm em không chỉ mô tả dataset mà còn phân tích kích thước ảnh, độ sáng, độ tương phản và thống kê màu để làm cơ sở chọn preprocessing và augmentation phù hợp.

---

## 7. `train_list.mat` và `test_list.mat` là gì?

Phải nhớ:

- là file annotation MATLAB
- không chứa ảnh
- chứa đường dẫn ảnh tương đối và nhãn lớp
- dùng để khôi phục official split

Câu nói:

> Hai file này là annotation gốc của Stanford Dogs, dùng để xác định official train/test split, chứ không phải thư mục ảnh.

---

## 8. `metadata_with_quality.csv` và `split_metadata.csv`

### `metadata_with_quality.csv`

- phục vụ EDA
- chứa metadata và quality stats

### `split_metadata.csv`

- phục vụ train/val/test pipeline
- ghi split cuối cùng

Câu nói:

> Nhóm em tách làm hai file để một file phục vụ phân tích dữ liệu đầu vào, còn một file phục vụ trực tiếp cho pipeline train/val/test.

---

## 9. Preprocessing cần nhớ

### Eval path

`Resize((256, 256)) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet)`

### Vì sao?

- chuẩn hóa kích thước
- đưa về input đúng cho pretrained model
- evaluation ổn định, deterministic

Câu nói:

> Nhóm em resize ảnh về 256 rồi center crop 224 để chuẩn hóa input cho cả ResNet-50 và ViT-B/16, đồng thời giữ evaluation path cố định và công bằng.

---

## 10. Normalization cần nhớ

- Công thức: `(x - mean) / std`
- Dùng ImageNet mean/std:
  - `(0.485, 0.456, 0.406)`
  - `(0.229, 0.224, 0.225)`

Lý do:

- model khởi tạo từ pretrained ImageNet
- cần khớp phân phối input mà checkpoint mong đợi

Câu nói:

> Nhóm em dùng ImageNet mean/std thay vì mean/std riêng của Stanford Dogs vì điều quan trọng hơn trong transfer learning là khớp với pretrained checkpoint.

---

## 11. Augmentation cần nhớ

### ResNet-50 mạnh hơn

- RandomResizedCrop rộng hơn
- flip
- rotation
- color jitter
- random erasing

### ViT-B/16 nhẹ hơn

- crop nhẹ hơn
- color jitter nhẹ hơn
- không rotation mạnh

Lý do:

- ViT trong bài fine-grained cần giữ breed cues tinh tế hơn

Câu nói:

> Nhóm em dùng augmentation mạnh hơn cho ResNet-50 và nhẹ hơn cho ViT-B/16 để tránh làm mất các tín hiệu fine-grained nhỏ quan trọng cho Vision Transformer.

---

## 12. Training strategy phải nhớ

Có 2 chiến lược:

- `Full fine-tuning for 12 epochs`
- `Head 3 epochs + full fine-tune 8 epochs`

Giải thích nhanh:

- full fine-tuning: mở toàn bộ model ngay từ đầu
- staged fine-tuning: học head trước, rồi mới mở toàn bộ backbone

Câu nói:

> Staged fine-tuning giúp phần classifier head thích nghi trước với bài toán 120 lớp mới, rồi sau đó mới tinh chỉnh toàn bộ backbone, nên thường ổn định hơn.

---

## 13. Hyperparameters chính

### Chung

- `CrossEntropyLoss`
- `AdamW`
- `CosineAnnealingLR`

### Learning rate

- ResNet-50 full: `1e-4`
- ResNet-50 staged: `1e-3 -> 1e-4`
- ViT-B/16 full: `3e-5`
- ViT-B/16 staged: `1e-3 -> 3e-5`

- weight decay: `1e-4`

---

## 14. Kết quả phải thuộc

| Model | Strategy | Accuracy | Macro F1 | ECE |
|---|---|---:|---:|---:|
| ResNet-50 | Full 12 | `85.57%` | `0.8485` | `0.0468` |
| ResNet-50 | Head 3 + Full 8 | `86.55%` | `0.8599` | `0.0408` |
| ViT-B/16 | Full 12 | `90.77%` | `0.9026` | `0.0178` |
| ViT-B/16 | Head 3 + Full 8 | `93.48%` | `0.9311` | `0.0198` |

### Câu kết quả quan trọng nhất

> Kết quả tốt nhất là ViT-B/16 với staged fine-tuning, đạt 93.48% accuracy và 0.9311 macro F1.

### Câu so sánh quan trọng nhất

> Staged strategy cải thiện kết quả cho cả ResNet-50 lẫn ViT-B/16, cho thấy head warmup trước khi full fine-tuning là hợp lý trên bài toán fine-grained này.

---

## 15. Cách diễn giải kết quả

### Vì sao ViT thắng?

- mạnh về mô hình hóa quan hệ toàn cục
- hợp với fine-grained classification
- pretrained mạnh
- hưởng lợi tốt từ staged fine-tuning

### Vì sao ResNet vẫn giá trị?

- nhẹ hơn nhiều
- nhanh hơn
- dễ deploy hơn
- dễ giải thích hơn với Grad-CAM

Câu nói:

> ViT-B/16 cho độ chính xác cao hơn, nhưng ResNet-50 vẫn là baseline rất có giá trị vì nhẹ hơn, nhanh hơn và thân thiện hơn cho các bối cảnh triển khai thực tế.

---

## 16. Interpretability nói sao cho an toàn

### ResNet-50

- dùng `Grad-CAM`
- thường dễ đọc hơn

### ViT-B/16

- dùng `attention rollout`
- là visualization định tính
- không phải bằng chứng nhân quả tuyệt đối

### Giải thích màu

- đỏ/vàng/cam: rollout mạnh hơn
- xanh/tím: rollout thấp hơn

Câu nói:

> Với ViT-B/16, nhóm em dùng attention rollout để quan sát vùng model nhấn mạnh tương đối nhiều hơn, nhưng đây chỉ là visualization định tính chứ không phải bằng chứng tuyệt đối về causal importance.

---

## 17. Calibration nói rất ngắn

- `ECE` càng thấp càng tốt
- ViT-B/16 tốt hơn ResNet-50 về calibration

Câu nói:

> Ngoài accuracy, nhóm em còn xem calibration bằng ECE, và ViT-B/16 cho độ tin cậy tốt hơn với ECE thấp hơn rõ rệt.

---

## 18. Demo Streamlit nói sao cho gọn

> Đây là app Streamlit nhóm em xây để demo trực tiếp 4 checkpoint benchmark. Người dùng có thể upload ảnh chó, chọn model, xem top prediction, confidence, và phần visualization như Grad-CAM hoặc attention rollout.

Nếu demo `ViT-B/16 staged`:

> Em chọn model tốt nhất là ViT-B/16 staged để minh họa khả năng dự đoán trên ảnh ngoài tập huấn luyện.

---

## 19. 3 câu chốt rất nên học thuộc

### Câu chốt 1

> Stanford Dogs là bài toán fine-grained phù hợp để so sánh CNN và Vision Transformer vì các lớp rất giống nhau và cần học các chi tiết tinh tế.

### Câu chốt 2

> Trong benchmark công bằng của nhóm em, ViT-B/16 staged đạt kết quả tốt nhất, còn ResNet-50 là baseline nhẹ và thực dụng hơn.

### Câu chốt 3

> Kết luận chính của phần image là Transformer mạnh hơn về hiệu năng trên bài toán này, nhưng CNN vẫn rất giá trị nếu xét đến chi phí và khả năng triển khai.

---

## 20. Nếu chỉ còn 30 giây trước khi lên nói, hãy nhớ 8 ý này

1. `Stanford Dogs`, `120 lớp`, `20,580 ảnh`
2. Fine-grained classification
3. Compare `ResNet-50` vs `ViT-B/16`
4. Split `10,200 / 1,800 / 8,580`
5. Input `224 x 224`, `batch_size = 32`
6. 2 strategies: full 12 và head 3 + full 8
7. Best result: `ViT-B/16 staged = 93.48%`
8. ResNet nhẹ hơn, ViT mạnh hơn
