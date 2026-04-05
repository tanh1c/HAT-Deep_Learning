# Hướng dẫn từ cơ bản đến bài làm Stanford Dogs:
# CNN, Transformer, ResNet-50, ViT-B/16, và cách gắn với Assignment 1

## 1. File này được viết để làm gì?

File này được viết cho người mới học AI, theo mục tiêu:

- bắt đầu từ những khái niệm cơ bản nhất
- giải thích rõ CNN là gì, Transformer là gì
- giải thích riêng 2 model bạn đang dùng: `ResNet-50` và `ViT-B/16`
- liên hệ trực tiếp đến bài image của bạn trong notebook:
  `assignments/assignment-1/image/notebooks/stanforddogs_resnet18_vit_report_workflow.ipynb`

Lưu ý nhỏ:

- Tên file notebook vẫn còn `resnet18` vì đây là tên cũ.
- Nội dung notebook hiện tại đã được cập nhật thành benchmark `ResNet-50` và `ViT-B/16` trên dataset `Stanford Dogs`.

Mục tiêu sau khi đọc xong file này:

- hiểu được một ảnh đi vào model như thế nào
- hiểu CNN và Transformer khác nhau ở đâu
- hiểu vì sao bài của bạn lại dùng `ResNet-50` và `ViT-B/16`
- hiểu preprocessing, augmentation, normalization, fine-tuning trong notebook
- hiểu vì sao kết quả cuối cùng của `ViT-B/16` tốt hơn `ResNet-50`

---

## 2. Trước hết: AI, Machine Learning, Deep Learning là gì?

### 2.1 AI là gì?

AI, hay Artificial Intelligence, là cách gọi rộng cho các hệ thống máy tính có thể làm những việc thường cần trí tuệ của con người, ví dụ:

- nhận diện ảnh
- dịch văn bản
- trả lời câu hỏi
- dự đoán nhãn

### 2.2 Machine Learning là gì?

Machine Learning là một nhánh của AI. Thay vì viết tay từng quy tắc, ta cho máy học từ dữ liệu.

Ví dụ:

- Nếu bạn muốn máy nhận ra “ảnh này là Chihuahua hay Bulldog”, bạn có thể đưa cho nó rất nhiều ảnh đã gán nhãn.
- Sau đó model tự học ra các mẫu hình trong dữ liệu để dự đoán cho ảnh mới.

### 2.3 Deep Learning là gì?

Deep Learning là một nhánh của Machine Learning, trong đó model được xây bằng các mạng neuron nhiều lớp.

Deep Learning rất mạnh ở:

- image classification
- speech recognition
- NLP
- multimodal learning

Trong bài của bạn, image track chính là một bài toán deep learning:

- input: ảnh chó
- output: tên giống chó trong 120 lớp

---

## 3. Bài toán image classification là gì?

Image classification là bài toán:

- đưa vào một bức ảnh
- model đưa ra nhãn của bức ảnh đó

Trong bài của bạn:

- input là ảnh chó
- output là 1 trong `120` giống chó của dataset `Stanford Dogs`

Ví dụ:

- ảnh 1 -> `Chihuahua`
- ảnh 2 -> `English Bulldog`
- ảnh 3 -> `Golden Retriever`

Đây là bài toán `fine-grained classification`, tức là phân lớp rất chi tiết.

Tại sao gọi là fine-grained?

- Vì không chỉ phân biệt “chó” và “mèo”
- Mà phải phân biệt giữa những giống chó rất giống nhau
- Model cần nhìn kỹ các chi tiết nhỏ như:
  - tai
  - mũi
  - bộ lông
  - họa tiết
  - tỉ lệ đầu và thân

Đây là lý do Stanford Dogs là dataset rất hợp để so sánh `CNN` và `ViT`.

---

## 4. Máy tính “nhìn” ảnh như thế nào?

Con người nhìn ảnh như một bức hình.
Máy tính thì nhìn ảnh như một tập hợp các con số.

### 4.1 Ảnh là ma trận pixel

Mỗi ảnh được tạo bởi rất nhiều pixel.

Ảnh màu RGB thường có 3 kênh:

- `R` = Red
- `G` = Green
- `B` = Blue

Nếu ảnh có kích thước `224 x 224`, thì có thể xem nó là tensor:

- `3 x 224 x 224`

Điều này có nghĩa:

- 3 kênh màu
- mỗi kênh là 1 ma trận `224 x 224`

### 4.2 Tensor là gì?

Tensor có thể hiểu đơn giản là:

- scalar: 1 con số
- vector: 1 dãy số
- matrix: 1 bảng 2 chiều
- tensor: mở rộng lên nhiều chiều hơn

Trong deep learning:

- 1 bức ảnh là 1 tensor
- 1 batch nhiều ảnh cùng lúc là 1 tensor lớn hơn

Trong notebook của bạn:

- `batch_size = 32`
- nên mỗi lần DataLoader đẩy vào model là 32 ảnh một lúc

Nếu input là ảnh màu `224 x 224`, 1 batch có thể có shape:

- `32 x 3 x 224 x 224`

---

## 5. Từ dữ liệu đến dự đoán: quy trình học cơ bản

Đây là luồng cơ bản nhất của một model học máy:

1. Nhập dữ liệu vào
2. Model dự đoán
3. So sánh dự đoán với nhãn đúng
4. Tính loss
5. Cập nhật trọng số để lần sau dự đoán tốt hơn

### 5.1 Loss là gì?

Loss là độ đo model đang sai bao nhiêu.

Trong bài của bạn, loss dùng là:

- `CrossEntropyLoss`

Nó rất phổ biến cho bài toán phân loại nhiều lớp.

Ý tưởng đơn giản:

- Nếu model đặt xác suất cao cho lớp đúng, loss sẽ thấp
- Nếu model đặt xác suất cao cho lớp sai, loss sẽ cao

### 5.2 Optimizer là gì?

Optimizer là bộ quy tắc giúp model cập nhật trọng số.

Trong notebook của bạn:

- `AdamW`

Đây là optimizer rất thông dụng cho transfer learning, CNN và Transformer.

### 5.3 Scheduler là gì?

Scheduler điều chỉnh learning rate theo thời gian.

Trong notebook của bạn:

- `CosineAnnealingLR`

Ý tưởng:

- lúc đầu cho model học nhanh hơn
- về sau giảm tốc để hội tụ ổn định hơn

---

## 6. CNN là gì?

CNN viết tắt của `Convolutional Neural Network`.

CNN được thiết kế đặc biệt cho dữ liệu dạng lưới, nhất là ảnh.

### 6.1 Ý tưởng cốt lõi của CNN

CNN không nhìn cả ảnh một lúc theo kiểu trải phẳng toàn bộ pixel.
Thay vào đó, nó nhìn các vùng nhỏ cục bộ và học đặc trưng từ các vùng này.

Nó rất hợp với ảnh vì:

- trong ảnh, các pixel gần nhau thường có liên quan
- các mẫu hình như cạnh, đường viền, texture, bộ lông, mắt, tai thường nằm ở các vùng cục bộ

### 6.2 Convolution là gì?

Convolution là phép quét một bộ lọc nhỏ trên ảnh.

Bạn có thể tưởng tượng:

- ảnh là 1 tấm vải
- filter là 1 con dấu nhỏ
- con dấu này đi quét qua ảnh
- mỗi vị trí nó rút ra 1 thông tin nào đó

Ví dụ:

- một filter có thể học để bắt đường viền dọc
- một filter khác học để bắt đường viền ngang
- một filter khác nữa học texture bộ lông

### 6.3 Ví dụ cực kỳ đơn giản

Giả sử ta có 1 filter muốn tìm đường viền dọc.
Nếu vùng ảnh bên trái tối, bên phải sáng, filter này sẽ phản ứng mạnh.

Từ đây, CNN có thể học:

- lớp đầu: cạnh, góc, texture
- lớp giữa: mắt, tai, chân, mũi
- lớp sau: đầu chó, thân chó, kiểu bộ lông, tổng thể giống chó

Đây là một điểm rất quan trọng:

- CNN học theo kiểu `từ đơn giản đến phức tạp`

### 6.4 Feature map là gì?

Sau khi filter quét qua ảnh, kết quả thu được là một `feature map`.

Có thể hiểu đơn giản:

- đây là bản đồ cho biết đặc trưng đó xuất hiện mạnh ở đâu

Nếu filter bắt “tai chó”, thì feature map có giá trị cao ở vùng có tai.

### 6.5 Stride và padding là gì?

`Stride`:

- là bước nhảy của filter khi quét ảnh
- stride lớn hơn -> feature map nhỏ hơn

`Padding`:

- là việc thêm viền xung quanh ảnh
- giúp giữ thông tin ở mép ảnh
- giúp kích thước không bị giảm quá nhanh

### 6.6 Activation function

Sau convolution, ta thường dùng activation như `ReLU`.

Mục đích:

- đưa tính phi tuyến vào model
- giúp model học được các mẫu phức tạp hơn

### 6.7 Pooling là gì?

Pooling là cách giảm kích thước feature map.

Ví dụ:

- `MaxPooling` lấy giá trị lớn nhất trong một ô nhỏ

Mục đích:

- giảm số tham số
- giảm chi phí tính toán
- giữ lại thông tin quan trọng

### 6.8 Tại sao CNN tốt cho image?

CNN có `inductive bias` rất hợp với ảnh, vì nó mặc định rằng:

- thông tin cục bộ quan trọng
- các pattern lặp lại có thể xuất hiện ở nhiều vị trí
- quan hệ không gian cần được khai thác theo kiểu cục bộ trước

Đây là lý do CNN rất mạnh và đã thống trị computer vision trong rất nhiều năm.

---

## 7. ResNet là gì và tại sao cần nó?

Khi CNN quá sâu, một vấn đề lớn xuất hiện:

- model khó học
- gradient có thể yếu dần hoặc học không ổn

ResNet được đề xuất để giải quyết vấn đề đó.

### 7.1 Vấn đề của mạng sâu

Trực giác thông thường là:

- mạng sâu hơn thì mạnh hơn

Nhưng trong thực tế:

- mạng quá sâu không dễ train
- thông tin và gradient khó đi xuyên qua rất nhiều lớp

### 7.2 Residual connection là gì?

Ý tưởng thiên tài của ResNet:

- thay vì học trực tiếp một hàm mới, model học phần `sửa đổi` so với đầu vào

Công thức trực giác:

- output = input + phần model học thêm

Nó giống như:

- Nếu đầu vào đã khá tốt, model chỉ cần học “chỉnh thêm một chút”
- không cần học lại tất cả từ đầu

### 7.3 Shortcut connection

ResNet dùng `shortcut` hay `skip connection`.

Nó cho phép thông tin:

- đi qua một số lớp nhanh hơn
- gradient quay ngược lại dễ hơn

Vì vậy:

- model sâu hơn nhưng vẫn train được tốt

---

## 8. ResNet-50 là gì?

`ResNet-50` là một phiên bản ResNet có độ sâu 50 lớp.

### 8.1 Tại sao gọi là 50?

Vì tổng số lớp học được xấp xỉ 50 lớp.

Nó sâu hơn rất nhiều so với các CNN cổ điển ban đầu.

### 8.2 Cấu trúc tổng quát của ResNet-50

Bạn có thể hình dung luồng đi như sau:

1. Ảnh đầu vào
2. Convolution đầu tiên
3. Nhiều cụm residual blocks
4. Global average pooling
5. Fully connected layer
6. Logits cho các lớp

### 8.3 Bottleneck block là gì?

ResNet-50 dùng `bottleneck block`.

Ý tưởng:

- giảm số kênh
- xử lý
- rồi mở rộng lại

Nó giúp:

- tiết kiệm tính toán
- vẫn giữ được khả năng biểu diễn mạnh

### 8.4 Global Average Pooling

Trước lớp phân loại cuối, ResNet-50 dùng `global average pooling`.

Có thể hiểu đơn giản:

- mỗi feature map được tóm tắt thành 1 số

Kết quả là một vector đặc trưng gọn hơn.

Trong ResNet-50 chuẩn:

- vector đặc trưng trước head có kích thước `2048`

Nếu là bài toán ImageNet:

- head cuối là `2048 -> 1000`

Trong bài của bạn:

- bài toán có `120` lớp
- nên head được thay thành `2048 -> 120`

### 8.5 Ví dụ gắn với bài Stanford Dogs

Giả sử model nhìn 1 ảnh Chihuahua:

- lớp sớm có thể phát hiện viền tai, viền mắt, texture lông
- lớp giữa có thể nhận ra tai nhọn, đầu nhỏ, mũi nhỏ
- lớp sau tổng hợp thành đặc trưng giống Chihuahua
- classifier head đưa ra xác suất cho 120 giống

### 8.6 Điểm mạnh của ResNet-50

- rất vững vàng
- dễ fine-tune
- tốt với dữ liệu image
- nhẹ hơn ViT-B/16 trong bài của bạn
- giải thích bằng Grad-CAM thường dễ đọc hơn

### 8.7 Điểm yếu của ResNet-50

- thiên về đặc trưng cục bộ
- có thể kém hơn Transformer trong một số bài toán fine-grained khi cần tổng hợp quan hệ toàn cục tốt hơn

---

## 9. Transformer là gì?

Transformer ban đầu nổi tiếng trong NLP.
Sau đó người ta nhận ra ý tưởng của nó cũng có thể áp dụng rất mạnh cho ảnh.

### 9.1 Tư duy khác với CNN

CNN:

- ưu tiên thông tin cục bộ
- học pattern bằng filter quét trên ảnh

Transformer:

- mô hình hóa quan hệ giữa các phần của đầu vào thông qua `attention`
- một phần của dữ liệu có thể “nhìn” một phần khác rất xa

Nó giống như:

- CNN nhìn ảnh theo kiểu “soi vào từng vùng”
- Transformer nhìn ảnh theo kiểu “phần này cần liên hệ với phần nào khác trong bức ảnh?”

### 9.2 Attention là gì?

Attention là cơ chế giúp model quyết định:

- nên chú ý nhiều vào phần nào
- phần nào liên quan đến phần nào

Trong NLP, một từ có thể cần nhìn các từ khác trong câu.

Trong image, một patch của ảnh có thể cần nhìn:

- patch ở đầu con chó
- patch ở tai
- patch ở lưng
- patch ở background

để tổng hợp thông tin.

### 9.3 Q, K, V là gì?

Đây là bộ ba nổi tiếng của attention:

- `Query`
- `Key`
- `Value`

Trực giác dễ hiểu:

- Query: “tôi đang tìm thông tin gì?”
- Key: “tôi có loại thông tin nào?”
- Value: “nếu bạn cần tôi, đây là thông tin tôi đưa cho bạn”

Model sẽ đo mức độ phù hợp giữa Query và Key, rồi lấy tổng có trọng số của các Value.

### 9.4 Self-attention

Self-attention có nghĩa là:

- các phần của cùng một input sẽ chú ý lẫn nhau

Trong image:

- patch vùng mắt có thể tương tác với patch vùng tai
- patch vùng ngực có thể tương tác với patch vùng thân

### 9.5 Multi-head attention

Transformer không chỉ có 1 attention.
Nó có nhiều “head”.

Mỗi head có thể học một kiểu quan hệ khác nhau:

- head này nhìn hình dáng
- head kia nhìn texture
- head khác nhìn bố cục tổng thể

### 9.6 Transformer block

Một Transformer block thường có:

1. Multi-head self-attention
2. Cộng residual
3. Layer normalization
4. MLP
5. Cộng residual tiếp

Bạn sẽ thấy ý tưởng residual ở đây rất giống tinh thần của ResNet:

- thông tin có đường đi tắt
- model dễ học hơn

---

## 10. ViT là gì?

ViT là `Vision Transformer`.

Ý tưởng lớn:

- biến ảnh thành một chuỗi patch
- xử lý chuỗi patch đó bằng Transformer

Nó giống như cách NLP xử lý chuỗi token.

### 10.1 Patch là gì?

Thay vì nhìn từng pixel riêng lẻ, ViT cắt ảnh thành các miền nhỏ đều nhau.

Trong `ViT-B/16`:

- ảnh được cắt thành patch `16 x 16`

Nếu ảnh có kích thước `224 x 224`, thì:

- mỗi chiều có `224 / 16 = 14` patch
- tổng cộng `14 x 14 = 196` patch

### 10.2 Mỗi patch trở thành 1 token

Mỗi patch `16 x 16 x 3` được trải phẳng thành vector.

Số phần tử ban đầu:

- `16 x 16 x 3 = 768`

Sau đó patch được chiếu vào không gian embedding của model.

### 10.3 CLS token là gì?

ViT thường thêm một token đặc biệt gọi là `CLS token`.

Có thể hiểu đơn giản:

- đây là “đại diện tổng hợp” cho cả ảnh

Sau nhiều lớp Transformer:

- thông tin từ tất cả patch sẽ được tổng hợp vào CLS token

Cuối cùng:

- classification head đọc CLS token để ra dự đoán

### 10.4 Positional embedding

Transformer không tự biết patch nào ở trên, dưới, trái, phải.

Vì vậy cần thêm `positional embedding` để model biết:

- patch này nằm ở đâu trong bức ảnh

Nếu không, model sẽ khó hiểu cấu trúc không gian của ảnh.

---

## 11. ViT-B/16 là gì?

`ViT-B/16` có thể tách nghĩa:

- `ViT` = Vision Transformer
- `B` = Base
- `16` = patch size 16

Đây là một model ViT có kích thước trung bình, rất phổ biến trong benchmark vision.

### 11.1 Luồng đi của ViT-B/16

1. Ảnh đầu vào `224 x 224`
2. Cắt thành 196 patch
3. Chuyển mỗi patch thành embedding
4. Thêm CLS token
5. Cộng positional embedding
6. Đưa qua nhiều Transformer encoder blocks
7. Lấy vector CLS cuối cùng
8. Đưa qua classifier head
9. Ra `120` logits trong bài của bạn

### 11.2 Kích thước head trong bài của bạn

Trong ViT-B/16 chuẩn:

- embedding dimension là `768`

Nên trong bài của bạn:

- classifier head được thay thành `768 -> 120`

### 11.3 Điểm mạnh của ViT-B/16

- mô hình hóa quan hệ toàn cục tốt
- mạnh trong nhiều bài toán fine-grained
- khi pretrained tốt, có thể cho kết quả rất cao

### 11.4 Điểm yếu của ViT-B/16

- nặng hơn ResNet-50
- tốn bộ nhớ hơn
- giải thích trực quan thường khó hơn CNN
- nếu train từ đầu trên dữ liệu nhỏ thì có thể không ổn bằng CNN

---

## 12. CNN và Transformer khác nhau ở đâu?

Bảng dưới đây là cách so sánh trực giác:

| Khía cạnh | CNN | Transformer / ViT |
|---|---|---|
| Cách nhìn ảnh | quét cục bộ bằng filter | chia patch và dùng attention |
| Inductive bias | mạnh, rất hợp với image | ít hơn, linh hoạt hơn |
| Khả năng gom thông tin xa | tăng dần theo nhiều lớp | rất tự nhiên qua attention |
| Tính ổn định khi fine-tune | thường rất tốt | tốt nếu pretrained mạnh |
| Dễ giải thích | thường dễ hơn với Grad-CAM | khó hơn, attention không phải lúc nào cũng dễ đọc |
| Chi phí tính toán | thường rẻ hơn | thường nặng hơn |

Nói ngắn gọn:

- CNN giống như chuyên gia soi từng chi tiết gần nhau
- Transformer giống như một hệ thống có thể xem mọi miền trong bức ảnh có liên quan đến miền nào khác

---

## 13. Tại sao bài của bạn dùng cả ResNet-50 và ViT-B/16?

Vì assignment yêu cầu so sánh:

- `CNN`
- và `Vision Transformer`

Trong bài của bạn:

- `ResNet-50` đại diện cho họ CNN
- `ViT-B/16` đại diện cho họ Transformer

Đây là so sánh hợp lý vì:

- cả hai đều là backbone mạnh
- cả hai đều có pretrained ImageNet
- cả hai đều có thể fine-tune cho bài 120 lớp

Notebook của bạn cũng có protocol công bằng:

- cùng dataset
- cùng split
- cùng input size
- cùng batch size
- cùng 2 chiến lược train

Điều đó giúp kết quả so sánh có ý nghĩa hơn.

---

## 14. Dataset Stanford Dogs trong bài của bạn là gì?

`Stanford Dogs` là dataset phân loại giống chó nổi tiếng.

Theo workflow của bạn:

- tổng số ảnh: `20,580`
- tổng số lớp: `120`
- official train split: `12,000`
- official test split: `8,580`

Sau đó pipeline của bạn tạo thêm validation từ official train:

- train: `10,200`
- val: `1,800`
- test: `8,580`

Số batch với `batch_size = 32`:

- train: `319`
- val: `57`
- test: `269`

### 14.1 `train_list.mat` và `test_list.mat` là gì?

Đây là 2 file annotation gốc của dataset.

Chúng không chứa pixel ảnh, mà chứa:

- đường dẫn ảnh tương đối
- nhãn lớp
- thông tin để xác định official split

Workflow của bạn đọc 2 file này để:

- khôi phục đúng split chính thức của Stanford Dogs

### 14.2 `metadata_with_quality.csv` và `split_metadata.csv` là gì?

`metadata_with_quality.csv`:

- bảng metadata theo từng ảnh
- có geometry, thống kê màu, độ sáng, contrast, saturation
- phục vụ EDA

`split_metadata.csv`:

- bảng split cuối cùng train/val/test
- phục vụ cho DataLoader và train/eval

Nếu nói ngắn gọn:

- một file để “hiểu dữ liệu”
- một file để “vận hành pipeline”

---

## 15. Preprocessing trong bài của bạn đang làm gì?

### 15.1 Vì sao phải resize và crop?

Ảnh trong Stanford Dogs có kích thước rất khác nhau.
Model thì cần input có kích thước cố định.

Trong bài của bạn, evaluation path dùng:

`Resize((256, 256)) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet)`

Ý nghĩa:

- `Resize((256, 256))`
  - đưa ảnh về kích thước nhất quán
- `CenterCrop(224)`
  - cắt vùng giữa để lấy input chuẩn cho model
- `ToTensor()`
  - chuyển ảnh sang tensor cho PyTorch
- `Normalize(ImageNet)`
  - đưa dữ liệu về đúng thang đo mà checkpoint pretrained mong đợi

### 15.2 Vì sao là 224?

Cả `ResNet-50` và `ViT-B/16` pretrained thường được thiết kế để làm việc tốt với input `224 x 224`.

Với `ViT-B/16`, kích thước này còn rất tiện:

- `224 / 16 = 14`
- nên ảnh chia thành `14 x 14 = 196` patch rất gọn đẹp

### 15.3 `ToTensor()` thực sự làm gì?

Nó:

- đổi ảnh từ dạng image/PIL sang tensor
- sắp xếp kênh theo kiểu deep learning
- thường đưa giá trị pixel từ khoảng `0-255` về khoảng `0-1`

---

## 16. Normalization là gì, và vì sao dùng ImageNet mean/std?

Công thức:

`x_norm = (x - mean) / std`

Trong bài của bạn:

- `mean = (0.485, 0.456, 0.406)`
- `std = (0.229, 0.224, 0.225)`

Đây là thông số ImageNet.

### 16.1 Vì sao không dùng mean/std của riêng Stanford Dogs?

Vì 2 model của bạn đều khởi tạo từ checkpoint pretrained trên ImageNet.

Checkpoint đó “quen” với kiểu input đã được normalize theo ImageNet.

Nên trong transfer learning, quan trọng hơn là:

- đưa input về đúng kiểu mà pretrained weights mong đợi

thay vì:

- bắt model phải làm quen lại với một quy ước scale mới

### 16.2 Ví dụ trực giác

Nếu model đã quen đọc ảnh được chuẩn hóa theo quy tắc A,
mà bạn đưa vào ảnh theo quy tắc B,
thì ở các lớp đầu tiên, thống kê đầu vào sẽ bị lệch.

Nó giống như:

- một người quen đọc nhiệt độ theo độ C
- bạn lại đưa bảng độ F

Không phải là không đọc được, nhưng sẽ dễ nhầm hơn.

---

## 17. Data augmentation là gì?

Data augmentation là cách tạo ra các biến thể hợp lý của ảnh train để model học mạnh hơn.

Ví dụ:

- crop khác nhau
- lật ngang
- xoay nhẹ
- đổi sáng/tối một chút
- che một phần nhỏ

Mục đích:

- model bớt học thuộc lòng
- robust hơn với dữ liệu thực tế

### 17.1 Augmentation của ResNet-50 trong notebook

Theo workflow của bạn, train transform của CNN là:

- `Resize((256, 256))`
- `RandomResizedCrop(224, scale=(0.72, 1.0))`
- `RandomHorizontalFlip(0.5)`
- `RandomRotation(15)`
- `ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)`
- `ToTensor()`
- `Normalize(ImageNet)`
- `RandomErasing(p=0.10)`

### 17.2 Augmentation của ViT-B/16 trong notebook

Train transform của ViT là:

- `Resize((256, 256))`
- `RandomResizedCrop(224, scale=(0.78, 1.0))`
- `RandomHorizontalFlip(0.5)`
- `ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10)`
- `ToTensor()`
- `Normalize(ImageNet)`
- `RandomErasing(p=0.08)`

### 17.3 Tại sao augmentation của ViT nhẹ hơn CNN?

Đây là câu hỏi rất hay.

Lý do trực giác:

- `ResNet-50` có inductive bias mạnh cho image, nên thường “chịu” augmentation khá tốt
- `ViT-B/16` trong bài fine-grained cần giữ các chi tiết giống chó khá tinh tế
- augmentation quá mạnh có thể làm méo các dấu hiệu nhỏ quan trọng

Vì vậy trong workflow của bạn:

- CNN được augmentation mạnh hơn
- ViT được augmentation nhẹ hơn để giữ breed cues

---

## 18. Transfer learning là gì?

Transfer learning là:

- lấy một model đã học trên dataset lớn
- rồi chuyển sang bài toán mới

Trong bài của bạn:

- cả `ResNet-50` và `ViT-B/16` đều bắt đầu từ pretrained `ImageNet`
- sau đó mới fine-tune trên `Stanford Dogs`

Tại sao cách này tốt?

- dataset ImageNet cực lớn
- model đã học được nhiều đặc trưng cơ bản của image
- bạn không cần train từ đầu
- hội tụ nhanh hơn, kết quả thường tốt hơn

---

## 19. Full fine-tuning và staged fine-tuning là gì?

Notebook của bạn dùng 2 chiến lược:

- `Full fine-tuning for 12 epochs`
- `Head 3 epochs + full fine-tune 8 epochs`

### 19.1 Full fine-tuning

Tất cả tham số của model cùng được học.

Ưu điểm:

- model có thể thích nghi mạnh

Nhược điểm:

- nếu learning rate không hợp lý, dễ làm hỏng tri thức pretrained

### 19.2 Staged fine-tuning

Giai đoạn 1:

- đóng băng backbone
- chỉ học classifier head

Giai đoạn 2:

- mở toàn bộ model
- fine-tune tiếp

Trực giác:

- đầu tiên cho “lớp cuối” học cách gán nhãn mới
- sau đó mới cho cả backbone điều chỉnh tinh tế

Đây thường ổn định hơn cho transfer learning.

Và đúng như kết quả bài của bạn, staged strategy cho kết quả tốt hơn cho cả hai family.

---

## 20. Hyperparameters cụ thể trong notebook của bạn

Đây là những con số rất nên nhớ vì nó gắn trực tiếp với bài làm.

### 20.1 Cấu hình chung

- `IMAGE_SIZE = 224`
- `BATCH_SIZE = 32`
- `FULL_FINETUNE_EPOCHS = 12`
- `HEAD_WARMUP_EPOCHS = 3`
- `HEAD_FINETUNE_EPOCHS = 8`

### 20.2 ResNet-50

- full fine-tuning:
  - `lr = 1e-4`
  - `weight_decay = 1e-4`
- staged:
  - head-only:
    - `lr = 1e-3`
  - full fine-tune:
    - `lr = 1e-4`
  - `weight_decay = 1e-4`

### 20.3 ViT-B/16

- full fine-tuning:
  - `lr = 3e-5`
  - `weight_decay = 1e-4`
- staged:
  - head-only:
    - `lr = 1e-3`
  - full fine-tune:
    - `lr = 3e-5`
  - `weight_decay = 1e-4`

### 20.4 Loss, optimizer, scheduler

- loss:
  - `CrossEntropyLoss`
- optimizer:
  - `AdamW`
- scheduler:
  - `CosineAnnealingLR`

---

## 21. ResNet-50 và ViT-B/16 được “đọ đầu” như thế nào trong bài của bạn?

Protocol của bạn rất quan trọng vì nó làm cho việc so sánh công bằng hơn.

Cả hai model cùng:

- dùng cùng dataset `Stanford Dogs`
- dùng cùng split `10,200 / 1,800 / 8,580`
- dùng cùng `batch_size = 32`
- dùng input `224 x 224`
- dùng 2 chiến lược train giống nhau về mặt logic
- được đánh giá bằng cùng metric

Metric chính:

- `Accuracy`
- `Macro F1`
- `ECE`

`ECE` là Expected Calibration Error.
Nó đo xem độ tự tin của model có “trung thực” không.

Ví dụ:

- Nếu model nói “tôi tự tin 90%” rất nhiều lần
- thì thực tế nó có đúng khoảng 90% không?

Nếu có, calibration tốt.
Nếu không, calibration kém.

---

## 22. Kết quả cuối cùng của bài image

Theo benchmark cuối cùng trong workflow/report của bạn:

| Model | Strategy | Params | Test Accuracy | Macro F1 | ECE |
|---|---|---:|---:|---:|---:|
| ResNet-50 | Full fine-tuning (12 epochs) | 23.75M | 85.57% | 0.8485 | 0.046781 |
| ResNet-50 | Head 3 + full 8 epochs | 23.75M | 86.55% | 0.8599 | 0.040777 |
| ViT-B/16 | Full fine-tuning (12 epochs) | 85.89M | 90.77% | 0.9026 | 0.017776 |
| ViT-B/16 | Head 3 + full 8 epochs | 85.89M | 93.48% | 0.9311 | 0.019799 |

### 22.1 Đọc bảng này thế nào?

`ViT-B/16 staged` là model tốt nhất về:

- accuracy
- macro F1

`ResNet-50 staged` là CNN tốt nhất của bạn.

Cả hai cùng cho thấy:

- staged fine-tuning tốt hơn full fine-tuning

### 22.2 Một kết luận quan trọng

`ViT-B/16` mạnh hơn về kết quả,
nhưng `ResNet-50` vẫn có giá trị vì:

- nhẹ hơn
- dễ train hơn
- dễ giải thích hơn
- dễ deploy hơn trong nhiều tình huống

---

## 23. Vì sao ViT-B/16 lại thắng trong bài này?

Không có 1 lý do duy nhất, nhưng có nhiều yếu tố hợp lại:

### 23.1 Bài toán fine-grained cần tổng hợp chi tiết nhỏ và quan hệ tổng thể

Để phân biệt các giống chó, model không chỉ cần texture cục bộ.
Nó cũng cần:

- quan hệ giữa mắt, tai, đầu, thân
- bố cục tổng thể của con chó

ViT thường làm việc này rất tốt nhiều khi.

### 23.2 ViT-B/16 được pretrained mạnh

Khi bắt đầu từ checkpoint ImageNet tốt:

- ViT có thể phát huy rất mạnh trên transfer learning

### 23.3 Staged fine-tuning giúp ổn định hơn

Cả hai model đều được lợi từ staged strategy.
Nhưng ViT có vẻ hưởng lợi rất rõ trên bài này.

### 23.4 Dataset đủ lớn để transfer learning có ý nghĩa

Stanford Dogs có hơn `20k` ảnh, trong đó train side vẫn khá lớn.
Điều này đủ cho fine-tuning pretrained models một cách nghiêm túc.

---

## 24. Tại sao ResNet-50 vẫn quan trọng dù được accuracy thấp hơn?

Rất nhiều người mới học hay nghĩ:

- model nào accuracy cao hơn thì model đó “tốt hơn tuyệt đối”

Thực ra không phải lúc nào cũng vậy.

ResNet-50 vẫn rất đáng giá vì:

- chỉ `23.75M` tham số, nhẹ hơn rất nhiều so với `85.89M` của ViT-B/16
- nhanh hơn
- ít tốn tài nguyên hơn
- Grad-CAM thường dễ đọc hơn
- dễ làm baseline và dễ deploy thực tế hơn

Nếu bài toán cần:

- model gọn
- chi phí rẻ
- thời gian nhanh

thì ResNet-50 vẫn là lựa chọn rất tốt.

---

## 25. Interpretability trong bài của bạn nên hiểu thế nào?

### 25.1 Grad-CAM cho ResNet-50

Grad-CAM tạo heatmap cho biết:

- vùng nào trong ảnh ảnh hưởng mạnh đến dự đoán của CNN

Nó thường dễ đọc hơn cho CNN vì CNN học đặc trưng không gian rõ ràng hơn.

### 25.2 Attention rollout cho ViT-B/16

Trong app demo của bạn, phần ViT đã được sửa thành `attention rollout`.

Nó không nên bị hiểu là:

- “đây là bằng chứng tuyệt đối rằng model chỉ nhìn vùng này”

Đúng hơn là:

- đây là một visualization định tính cho thấy luồng attention tổng hợp qua nhiều layer

Ý nghĩa màu sắc:

- đỏ/vàng/cam: mức rollout cao hơn
- xanh/tím: mức rollout thấp hơn

Trong `overlay`:

- heatmap được chồng lên ảnh gốc
- vùng nóng hơn là nơi model nhấn mạnh tương đối nhiều hơn trong visualization đó

Nhưng cần cẩn thận:

- nó không phải xác suất
- nó không phải bằng chứng nhân quả tuyệt đối

---

## 26. Nếu phải giải thích cho một người chưa học AI, nên nói ngắn gọn thế nào?

Bạn có thể nói:

> Trong bài này, chúng em so sánh hai cách “nhìn ảnh”.
> CNN, cụ thể là ResNet-50, nhìn ảnh bằng cách quét các vùng nhỏ và học dần từ chi tiết đến tổng thể.
> Transformer, cụ thể là ViT-B/16, cắt ảnh thành các patch rồi dùng attention để mô hình hóa quan hệ giữa các phần của ảnh.
> Trên dataset Stanford Dogs, ViT-B/16 đạt kết quả cao hơn, nhưng ResNet-50 vẫn là baseline tốt vì nhẹ và dễ giải thích hơn.

---

## 27. Ví dụ cực kỳ gần với bài của bạn: một ảnh Chihuahua

Giả sử có 1 ảnh Chihuahua nằm trên đệm.

### 27.1 ResNet-50 có thể học như sau

- lớp đầu bắt viền tai
- lớp giữa bắt đầu nhỏ, tai nhọn, mũi nhỏ
- lớp sau tổng hợp thành “kiểu Chihuahua”
- head 120 lớp chọn ra lớp Chihuahua

### 27.2 ViT-B/16 có thể học như sau

- chia ảnh thành 196 patch
- patch vùng tai, mắt, mũi, thân và cả nền cùng được đưa vào encoder
- attention giúp model liên hệ patch mắt với patch tai và hình dáng tổng thể
- CLS token tổng hợp toàn bộ thông tin
- head 120 lớp dự đoán Chihuahua

### 27.3 Tại sao đây là fine-grained?

Vì model có thể bị nhầm với các giống:

- tai nhọn
- đầu nhỏ
- thân gọn

nên chi tiết rất quan trọng.

---

## 28. Những hiểu nhầm thường gặp của người mới học

### Hiểu nhầm 1: Transformer lúc nào cũng hơn CNN

Không đúng.

Kết quả phụ thuộc vào:

- dữ liệu
- quy mô dữ liệu
- pretraining
- tuning
- tài nguyên tính toán

### Hiểu nhầm 2: Attention map là bằng chứng giải thích hoàn hảo

Không đúng.

Attention visualization hữu ích, nhưng không phải lúc nào cũng đồng nghĩa với causal importance.

### Hiểu nhầm 3: Accuracy là tất cả

Không đúng.

Bạn còn cần xem:

- Macro F1
- calibration
- chi phí tính toán
- tính dễ deploy

### Hiểu nhầm 4: Model có nhiều tham số hơn thì chắc chắn tốt hơn

Không hoàn toàn.

Model lớn hơn có thể mạnh hơn, nhưng:

- nặng hơn
- tốn tài nguyên hơn
- cần tuning kỹ hơn

---

## 29. Nếu bạn muốn đọc notebook của mình với tư duy “hiểu mô hình”, nên đọc theo thứ tự nào?

Để đọc notebook `stanforddogs_resnet18_vit_report_workflow.ipynb`, bạn có thể đi theo luồng:

### Bước 1: Nhìn phần khai báo dataset và split

Tập trung vào:

- `train_list.mat`
- `test_list.mat`
- metadata
- split train/val/test

Mục tiêu:

- hiểu dữ liệu vào của model là gì

### Bước 2: Nhìn phần transform

Tập trung vào:

- `Resize`
- `RandomResizedCrop`
- `CenterCrop`
- `Normalize`

Mục tiêu:

- hiểu ảnh được chuẩn hóa như thế nào trước khi học

### Bước 3: Nhìn phần model builder

Tập trung vào:

- head của ResNet-50 được thay thế như thế nào
- head của ViT-B/16 được thay thế như thế nào

Mục tiêu:

- hiểu pretrained model đã được “đổi đầu ra” cho bài 120 lớp ra sao

### Bước 4: Nhìn phần train loop

Tập trung vào:

- `CrossEntropyLoss`
- `AdamW`
- `CosineAnnealingLR`
- logic `head_only` và `full_finetune`

Mục tiêu:

- hiểu model học như thế nào

### Bước 5: Nhìn phần evaluation

Tập trung vào:

- accuracy
- macro F1
- ECE
- confusion / calibration / interpretability

Mục tiêu:

- hiểu cách kết luận model nào tốt hơn

---

## 30. Một cách nhớ nhanh cho thuyết trình

Nếu cần nhớ thật gọn:

- `CNN / ResNet-50`:
  - nhìn ảnh theo vùng cục bộ
  - học từ cạnh -> bộ phận -> tổng thể
  - nhẹ hơn, dễ giải thích hơn

- `Transformer / ViT-B/16`:
  - cắt ảnh thành patch
  - dùng attention để liên hệ các patch
  - mạnh hơn trong benchmark Stanford Dogs của bạn

- `Bài của bạn`:
  - dataset `Stanford Dogs`
  - split `10,200 / 1,800 / 8,580`
  - compare `ResNet-50` vs `ViT-B/16`
  - staged fine-tuning tốt hơn full fine-tuning
  - best model: `ViT-B/16 staged = 93.48%`

---

## 31. Kết luận cuối cùng

Nếu phải tóm tắt toàn bộ file này trong vài câu:

- CNN và Transformer là hai cách khác nhau để máy tính “nhìn” ảnh.
- `ResNet-50` là một CNN mạnh, ổn định, nhẹ hơn, dễ giải thích hơn.
- `ViT-B/16` là một Vision Transformer chia ảnh thành patch và dùng attention để tổng hợp thông tin toàn cục.
- Trong bài `Stanford Dogs` của bạn, `ViT-B/16` cho kết quả cao nhất, nhưng `ResNet-50` vẫn là baseline rất có giá trị.
- Workflow trong notebook của bạn là một ví dụ rất tốt của transfer learning thực chiến: EDA, preprocessing, augmentation, fine-tuning, evaluation, calibration, và demo Streamlit.

---

## 32. Gợi ý học tiếp nếu bạn muốn đào sâu hơn

Nếu bạn muốn tiếp tục học sau file này, thứ tự hợp lý là:

1. Hiểu chắc `tensor`, `loss`, `optimizer`, `backpropagation`
2. Hiểu kỹ `convolution`, `feature map`, `pooling`
3. Hiểu `residual connection`
4. Hiểu `self-attention`, `Q-K-V`, `CLS token`
5. Đọc lại notebook của bạn và map từng phần code vào lý thuyết
6. Thử tự giải thích lại bài của mình bằng ngôn ngữ đơn giản cho người khác

Nếu bạn làm được điều đó, nghĩa là bạn đã không chỉ “chạy notebook”, mà đã thực sự bắt đầu hiểu mô hình.
