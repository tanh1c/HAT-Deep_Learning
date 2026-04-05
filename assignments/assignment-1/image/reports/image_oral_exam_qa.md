# Q&A Ôn Tập Trước Khi Thuyết Trình Phần Image

File này dành để ôn trước khi thuyết trình hoặc chuẩn bị cho phần hỏi đáp với giảng viên.

Mình chia theo 4 nhóm:

- kiến thức nền
- kiến trúc mô hình
- quyết định kỹ thuật trong bài
- kết quả, phân tích, và phản biện

---

## A. Kiến thức nền

### Q1. Image classification là gì?

Image classification là bài toán đưa một ảnh vào và dự đoán ảnh đó thuộc lớp nào.

Trong bài của bạn:

- input là ảnh chó
- output là 1 trong `120` giống chó

### Q2. Fine-grained classification là gì?

Fine-grained classification là phân loại giữa các lớp rất giống nhau về mặt thị giác.

Ví dụ:

- không phải chỉ phân biệt chó với mèo
- mà là phân biệt Chihuahua với các giống chó nhỏ khác

### Q3. Tensor là gì?

Tensor là cấu trúc dữ liệu nhiều chiều dùng để biểu diễn dữ liệu trong deep learning.

Ví dụ:

- một ảnh RGB `224 x 224` có thể được biểu diễn là `3 x 224 x 224`
- một batch 32 ảnh là `32 x 3 x 224 x 224`

### Q4. Loss là gì?

Loss là độ đo mức sai lệch giữa dự đoán của model và nhãn đúng.

Trong bài của bạn dùng:

- `CrossEntropyLoss`

### Q5. Optimizer là gì?

Optimizer là thuật toán cập nhật trọng số của model để giảm loss.

Trong bài của bạn dùng:

- `AdamW`

### Q6. Learning rate là gì?

Learning rate là tốc độ cập nhật trọng số.

- quá lớn: dễ dao động
- quá nhỏ: học chậm

### Q7. Scheduler là gì?

Scheduler điều chỉnh learning rate theo thời gian.

Trong bài của bạn dùng:

- `CosineAnnealingLR`

---

## B. CNN, Transformer, ResNet-50, ViT-B/16

### Q8. CNN là gì?

CNN là mạng neuron dùng convolution để học đặc trưng cục bộ từ ảnh.

Nó thường học theo thứ tự:

- cạnh, texture
- bộ phận
- đặc trưng tổng thể

### Q9. Convolution là gì?

Convolution là phép cho một bộ lọc nhỏ quét qua ảnh để phát hiện pattern như cạnh, đường viền, texture.

### Q10. Feature map là gì?

Feature map là đầu ra sau convolution, cho biết đặc trưng nào xuất hiện mạnh ở đâu trong ảnh.

### Q11. Pooling là gì?

Pooling là phép giảm kích thước feature map để giảm tính toán và giữ thông tin quan trọng.

### Q12. ResNet là gì?

ResNet là họ CNN dùng residual connection để giúp train mạng sâu tốt hơn.

### Q13. Residual connection là gì?

Residual connection là skip connection cho phép:

- thông tin đi tắt qua một số lớp
- gradient truyền tốt hơn

### Q14. ResNet-50 là gì?

ResNet-50 là một mạng ResNet sâu khoảng 50 lớp, rất phổ biến trong computer vision.

### Q15. Vì sao ResNet-50 phù hợp làm baseline CNN?

Vì nó:

- đủ mạnh
- khá ổn định khi fine-tune
- phổ biến
- có pretrained ImageNet
- nhẹ hơn ViT-B/16

### Q16. Transformer là gì?

Transformer là kiến trúc dùng attention để mô hình hóa quan hệ giữa các phần của dữ liệu.

### Q17. Attention là gì?

Attention là cơ chế cho model biết nên chú ý nhiều vào phần nào và phần nào liên quan đến phần nào.

### Q18. Self-attention là gì?

Self-attention là khi các phần của cùng một input tương tác với nhau để trao đổi thông tin.

### Q19. ViT là gì?

ViT là Vision Transformer, tức là áp dụng ý tưởng Transformer cho ảnh.

### Q20. ViT-B/16 nghĩa là gì?

- `ViT` = Vision Transformer
- `B` = Base
- `16` = patch size 16

### Q21. Patch là gì trong ViT?

Patch là một vùng nhỏ của ảnh được cắt ra để coi như một token.

Với `224 x 224` và patch size `16`:

- mỗi chiều có 14 patch
- tổng là 196 patch

### Q22. CLS token là gì?

CLS token là token đặc biệt dùng để tổng hợp thông tin toàn ảnh, sau đó đưa vào classifier head.

### Q23. CNN và ViT khác nhau cốt lõi ở đâu?

CNN:

- mạnh về đặc trưng cục bộ
- dùng convolution filter

ViT:

- chia ảnh thành patch
- dùng attention để gom quan hệ toàn cục

### Q24. Tại sao ViT thường mạnh trong fine-grained classification?

Vì nó có thể học quan hệ giữa nhiều vùng ảnh khác nhau tốt hơn, phù hợp khi phải phân biệt những khác biệt tinh tế giữa các lớp.

---

## C. Dataset Stanford Dogs và pipeline của bài bạn

### Q25. Vì sao chọn Stanford Dogs?

Vì:

- có `120` lớp
- đủ lớn
- là bài toán fine-grained thật sự
- phù hợp để so sánh CNN và ViT

### Q26. Stanford Dogs có bao nhiêu ảnh?

- tổng: `20,580`
- official train: `12,000`
- official test: `8,580`

### Q27. `train_list.mat` và `test_list.mat` là gì?

Đây là hai file annotation MATLAB dùng để khôi phục official split của dataset.

Chúng chứa:

- đường dẫn ảnh
- nhãn lớp

chứ không chứa pixel ảnh.

### Q28. Vì sao phải tạo validation split từ official train?

Vì cần một tập validation để:

- chọn checkpoint tốt nhất
- theo dõi quá trình train
- tune mô hình một cách hợp lý

### Q29. Split cuối cùng của bài bạn là gì?

- train: `10,200`
- val: `1,800`
- test: `8,580`

### Q30. Vì sao test set vẫn giữ nguyên official test?

Để đảm bảo benchmark khách quan và gần với protocol gốc của dataset.

### Q31. `metadata_with_quality.csv` dùng để làm gì?

Dùng cho EDA:

- phân tích kích thước ảnh
- độ sáng
- contrast
- saturation
- thống kê màu RGB

### Q32. `split_metadata.csv` dùng để làm gì?

Dùng để lưu split cuối cùng:

- train
- val
- test

và phục vụ pipeline DataLoader.

### Q33. Vì sao phải tách 2 file metadata?

Vì chúng phục vụ 2 nhiệm vụ khác nhau:

- một file cho phân tích dữ liệu
- một file cho vận hành pipeline train/eval

---

## D. Preprocessing và augmentation

### Q34. Vì sao phải resize ảnh?

Vì ảnh gốc có kích thước không đồng nhất, trong khi model cần input cố định.

### Q35. Vì sao dùng `Resize((256, 256)) -> CenterCrop(224)`?

Để:

- chuẩn hóa ảnh
- đưa input về đúng kích thước `224 x 224`
- giữ evaluation ổn định và deterministic

### Q36. Vì sao là `224 x 224`?

Vì đây là kích thước đầu vào chuẩn, phù hợp với pretrained ResNet-50 và ViT-B/16.

### Q37. `ToTensor()` làm gì?

Nó chuyển ảnh sang tensor và đưa dữ liệu về định dạng model có thể xử lý.

### Q38. Normalization là gì?

Normalization là chuẩn hóa dữ liệu đầu vào theo công thức:

`x_norm = (x - mean) / std`

### Q39. Vì sao dùng ImageNet mean/std?

Vì model bắt đầu từ checkpoint pretrained trên ImageNet, nên cần khớp thống kê đầu vào mà checkpoint mong đợi.

### Q40. Vì sao không dùng mean/std riêng của Stanford Dogs?

Có thể làm vậy trong một số bài toán, nhưng trong transfer learning với pretrained ImageNet, khớp với ImageNet normalization thường hợp lý hơn.

### Q41. Data augmentation là gì?

Là tạo các biến thể hợp lý của ảnh train để model học robust hơn.

### Q42. Trong bài của bạn, augmentation của ResNet-50 gồm gì?

- RandomResizedCrop
- HorizontalFlip
- Rotation
- ColorJitter
- RandomErasing

### Q43. Trong bài của bạn, augmentation của ViT-B/16 gồm gì?

- RandomResizedCrop
- HorizontalFlip
- ColorJitter nhẹ hơn
- RandomErasing nhẹ hơn

### Q44. Vì sao augmentation của ViT nhẹ hơn CNN?

Vì ViT trong bài fine-grained cần giữ các chi tiết giống chó tinh tế, nên augmentation quá mạnh có thể làm mất tín hiệu quan trọng.

---

## E. Training strategy và hyperparameters

### Q45. Transfer learning là gì?

Transfer learning là lấy model đã pretrained trên dataset lớn rồi fine-tune cho bài toán mới.

### Q46. Trong bài bạn pretrained từ đâu?

Từ pretrained ImageNet.

### Q47. Có mấy chiến lược training?

Có 2:

- full fine-tuning 12 epochs
- head 3 epochs + full fine-tune 8 epochs

### Q48. Full fine-tuning là gì?

Là mở toàn bộ model để học ngay từ đầu.

### Q49. Staged fine-tuning là gì?

Là:

- giai đoạn 1: chỉ train head
- giai đoạn 2: mở toàn bộ model để fine-tune

### Q50. Vì sao staged fine-tuning thường tốt hơn?

Vì nó giúp classifier head thích nghi trước với bài toán mới, sau đó backbone mới điều chỉnh tinh hơn, nên thường ổn định hơn.

### Q51. Batch size của bài bạn là bao nhiêu?

- `32`

### Q52. Loss của bài bạn là gì?

- `CrossEntropyLoss`

### Q53. Optimizer là gì?

- `AdamW`

### Q54. Scheduler là gì?

- `CosineAnnealingLR`

### Q55. Learning rate của ResNet-50 là gì?

- full: `1e-4`
- staged: `1e-3 -> 1e-4`

### Q56. Learning rate của ViT-B/16 là gì?

- full: `3e-5`
- staged: `1e-3 -> 3e-5`

### Q57. Weight decay là bao nhiêu?

- `1e-4`

---

## F. Kết quả và diễn giải

### Q58. Model nào tốt nhất?

`ViT-B/16` với staged fine-tuning.

### Q59. Accuracy tốt nhất là bao nhiêu?

- `93.48%`

### Q60. Macro F1 tốt nhất là bao nhiêu?

- `0.9311`

### Q61. ResNet-50 tốt nhất đạt bao nhiêu?

- `86.55%` accuracy
- `0.8599` macro F1

### Q62. Kết quả có cho thấy staged strategy hữu ích không?

Có. Nó cải thiện kết quả cho cả ResNet-50 lẫn ViT-B/16.

### Q63. Vì sao phải báo cả Macro F1 chứ không chỉ Accuracy?

Vì Macro F1 giúp nhìn công bằng hơn giữa các lớp, nhất là khi mức độ khó giữa các lớp khác nhau.

### Q64. ECE là gì?

ECE là Expected Calibration Error, đo mức độ đáng tin của confidence score.

### Q65. ECE thấp có nghĩa gì?

ECE thấp nghĩa là độ tự tin của model phù hợp hơn với độ chính xác thực tế.

### Q66. Model nào calibration tốt hơn?

ViT-B/16 tốt hơn vì ECE thấp hơn rõ rệt.

### Q67. Vì sao ViT-B/16 thắng ResNet-50?

Có thể tóm gọn:

- mô hình hóa quan hệ toàn cục tốt hơn
- phù hợp với fine-grained classification
- pretrained mạnh
- hưởng lợi tốt từ staged fine-tuning

### Q68. Vì sao ResNet-50 vẫn đáng giá?

Vì nó:

- nhẹ hơn
- nhanh hơn
- dễ triển khai hơn
- dễ giải thích hơn

---

## G. Interpretability, demo, và phản biện

### Q69. Grad-CAM là gì?

Grad-CAM là kỹ thuật tạo heatmap cho CNN để cho biết vùng nào ảnh hưởng mạnh đến dự đoán.

### Q70. Attention rollout là gì?

Attention rollout là cách tổng hợp attention qua nhiều layer của ViT để tạo visualization định tính về vùng model nhấn mạnh tương đối nhiều hơn.

### Q71. Màu đỏ/vàng trong attention rollout nghĩa là gì?

Nó cho thấy rollout value cao hơn, tức là attention flow tổng hợp mạnh hơn ở vùng đó.

### Q72. Overlay là gì?

Overlay là heatmap được chồng lên ảnh gốc để dễ nhìn vùng model nhấn mạnh.

### Q73. Attention rollout có phải bằng chứng tuyệt đối model đang “hiểu” đúng không?

Không. Đây là visualization định tính, hỗ trợ diễn giải chứ không phải bằng chứng nhân quả tuyệt đối.

### Q74. Nếu thầy hỏi “vì sao attention lại dính background” thì trả lời sao?

Có thể trả lời:

> Attention rollout chỉ là một visualization định tính. Nó cho thấy vùng có attention flow tổng hợp mạnh hơn, nhưng không phải lúc nào cũng đồng nghĩa với causal importance tuyệt đối. Vì vậy nhóm em dùng nó như công cụ hỗ trợ diễn giải, không xem nó là bằng chứng hoàn hảo.

### Q75. Demo Streamlit để làm gì?

Để:

- minh họa dự đoán trực tiếp
- cho người xem thử ảnh ngoài tập huấn luyện
- xem confidence
- xem interpretability artifacts

### Q76. Vì sao làm thêm Streamlit app?

Vì nó giúp bài làm:

- trực quan hơn
- dễ trình bày hơn
- có giá trị triển khai/demo thực tế hơn

---

## H. Câu hỏi phản biện nâng cao

### Q77. Nếu thầy hỏi “vì sao không train từ đầu?”

Có thể trả lời:

> Vì bài toán này phù hợp với transfer learning hơn. Dùng pretrained ImageNet giúp hội tụ nhanh hơn, ổn định hơn, và cho kết quả tốt hơn so với train from scratch trong giới hạn thời gian và tài nguyên của assignment.

### Q78. Nếu thầy hỏi “vì sao không dùng model lớn hơn?”

> Nhóm em ưu tiên một benchmark công bằng, thực tế và đủ mạnh. ResNet-50 và ViT-B/16 là hai backbone điển hình, phổ biến, có pretrained tốt và vẫn phù hợp với tài nguyên của bài tập lớn.

### Q79. Nếu thầy hỏi “vì sao không dùng custom mean/std?”

> Vì trong transfer learning với pretrained ImageNet, điều quan trọng là khớp đầu vào với phân phối mà checkpoint đã học trước đó. Dùng ImageNet normalization là lựa chọn hợp lý và ổn định hơn trong bối cảnh này.

### Q80. Nếu thầy hỏi “vì sao không chỉ dùng accuracy?”

> Vì accuracy chưa phản ánh hết hành vi của model. Nhóm em bổ sung Macro F1 để nhìn cân bằng hơn giữa các lớp, và ECE để đánh giá độ tin cậy của confidence score.

### Q81. Nếu thầy hỏi “đóng góp chính của phần image là gì?”

> Đóng góp chính là xây dựng một pipeline hoàn chỉnh cho Stanford Dogs, thực hiện benchmark công bằng giữa ResNet-50 và ViT-B/16 với hai chiến lược fine-tuning, và bổ sung thêm calibration, interpretability, cùng demo Streamlit để minh họa khả năng ứng dụng.

---

## I. 10 câu trả lời ngắn nên học thuộc

### 1.

> Đây là bài toán fine-grained classification trên Stanford Dogs với 120 lớp.

### 2.

> Nhóm em so sánh ResNet-50 và ViT-B/16 dưới cùng một protocol để đảm bảo công bằng.

### 3.

> Official split được khôi phục từ train_list.mat và test_list.mat.

### 4.

> Split cuối cùng là 10,200 train, 1,800 val, và 8,580 test.

### 5.

> Input được chuẩn hóa về 224 x 224 và normalize theo ImageNet mean/std.

### 6.

> Nhóm em dùng hai chiến lược là full fine-tuning và staged fine-tuning.

### 7.

> Kết quả tốt nhất là ViT-B/16 staged với 93.48% accuracy.

### 8.

> Staged fine-tuning cải thiện kết quả cho cả hai họ mô hình.

### 9.

> ViT-B/16 mạnh hơn về hiệu năng, còn ResNet-50 nhẹ hơn và dễ triển khai hơn.

### 10.

> Attention rollout và Grad-CAM được dùng như công cụ hỗ trợ diễn giải, không phải bằng chứng tuyệt đối.
