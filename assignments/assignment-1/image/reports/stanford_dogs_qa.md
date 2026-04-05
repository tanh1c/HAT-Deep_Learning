# Stanford Dogs Image Q&A

## Q1. `metadata_with_quality.csv` và `split_metadata.csv` khác nhau chỗ nào, vì sao phải tách làm 2 file riêng, công dụng của từng file là gì? Các phần phân tích input dataset ở node 1 (EDA) có tác dụng chính là gì khi sang node 2 là preprocessing?

### A1.

Hai file này liên quan với nhau nhưng không phục vụ cùng một mục đích.

`metadata_with_quality.csv` là file metadata tổng cho toàn bộ dataset sau khi notebook đã khôi phục lại official split của Stanford Dogs từ `train_list.mat` và `test_list.mat`, rồi làm thêm bước enrich bằng các thống kê chất lượng ảnh. File này thiên về **EDA và phân tích dữ liệu đầu vào**.

Nó thường chứa các nhóm thông tin như:

- định danh mẫu: `image_id`, `class_name`, `label`
- nguồn split gốc của dataset: `official_split`
- thông tin hình học: `width`, `height`, `aspect_ratio`
- thông tin chất lượng ảnh: `brightness_mean`, `contrast_std`, `saturation_mean`
- thống kê màu: `r_mean`, `g_mean`, `b_mean`

Nói ngắn gọn, `metadata_with_quality.csv` trả lời câu hỏi:

- dữ liệu đầu vào trông như thế nào?
- ảnh có kích thước ra sao?
- độ sáng, độ tương phản, màu sắc có phân bố thế nào?
- có sự lệch lớn nào giữa các ảnh hoặc giữa các breed hay không?

`split_metadata.csv` là file phục vụ **pipeline huấn luyện** sau khi notebook đã tách tiếp `official_train` thành `train` và `val`, rồi giữ nguyên `test`. File này thiên về **quản lý split cuối cùng để cấp dữ liệu cho Dataset/DataLoader**.

Nó thường giữ lại:

- định danh mẫu và class
- `official_split` để vẫn truy vết được split gốc
- cột `split` cuối cùng với các giá trị như `train`, `val`, `test`
- các trường hình học cơ bản cần cho việc nối metadata với pipeline

Nói ngắn gọn, `split_metadata.csv` trả lời câu hỏi:

- ảnh nào thuộc train?
- ảnh nào thuộc val?
- ảnh nào thuộc test?
- DataLoader phải đọc sample nào cho từng phase?

### Vì sao phải tách thành 2 file riêng?

Tách riêng giúp workflow rõ ràng hơn và tránh trộn hai nhiệm vụ khác nhau vào một file:

- `metadata_with_quality.csv` phục vụ phân tích dữ liệu đầu vào và trực quan hóa ở node 1
- `split_metadata.csv` phục vụ build dataset, build dataloader, và training/evaluation ở node 2 trở đi

Nếu chỉ giữ một file duy nhất thì sẽ có vài bất tiện:

- file dễ bị “quá tải vai trò”, vừa để EDA vừa để feed training pipeline
- khi thay đổi cách split nội bộ `train/val`, bạn phải cập nhật lại cả file phân tích tổng
- khó giữ ranh giới rõ giữa “official dataset description” và “final experimental split”

Tách làm 2 file giúp:

- tái sử dụng EDA ổn định hơn
- thay đổi split mà không làm rối phần mô tả dataset gốc
- dễ giải thích trong report theo đúng flow `node 1 = input/EDA`, `node 2 = preprocessing/loading`

### Công dụng chính của từng file

`metadata_with_quality.csv`:

- làm nguồn cho các biểu đồ EDA
- phân tích phân bố kích thước ảnh
- phân tích độ sáng, tương phản, saturation
- kiểm tra RGB mean theo toàn dataset hoặc theo breed
- hỗ trợ quyết định preprocessing phù hợp

`split_metadata.csv`:

- làm nguồn cho `train / val / test` split cuối cùng
- giúp `Dataset` tra đúng ảnh và nhãn
- giúp `DataLoader` lấy đúng sample theo phase
- làm cầu nối giữa metadata và training pipeline thực tế

### Vai trò chính của node 1 (EDA) trước khi sang node 2 (preprocessing)

Node 1 không chỉ để “mô tả dataset cho đẹp”, mà có vai trò kỹ thuật rất rõ: nó giúp ta hiểu đặc điểm đầu vào trước khi thiết kế preprocessing.

EDA ở node 1 giúp trả lời các câu hỏi nền tảng như:

- ảnh có đồng nhất kích thước hay không?
- aspect ratio có biến thiên lớn không?
- ảnh có quá tối hoặc quá sáng nhiều không?
- phân bố màu có lệch không?
- số lượng ảnh giữa các lớp có cân bằng tương đối không?
- có điểm nào cần lưu ý để tránh augmentation hoặc normalization quá mạnh không?

Từ các hiểu biết đó, node 2 mới có cơ sở để chọn preprocessing hợp lý:

- chọn `Resize` và `CenterCrop` để chuẩn hóa kích thước đầu vào
- quyết định augmentation nào nên dùng hay nên hạn chế
- chọn normalization phù hợp với pretrained models
- cấu hình dataset/dataloader ổn định cho train, val, test

### Tóm tắt ngắn

- `metadata_with_quality.csv` = file cho EDA, phân tích đặc điểm và chất lượng ảnh đầu vào
- `split_metadata.csv` = file cho split cuối cùng, phục vụ dataset/dataloader và training pipeline
- node 1 giúp hiểu dữ liệu đầu vào
- node 2 dùng hiểu biết đó để thiết kế preprocessing đúng và thực tế hơn

## Q2. Ở node 2, trong quá trình Image resizing and standardization, tại sao lại dùng `Resize((256, 256)) -> CenterCrop(224) -> ToTensor()`? Bản chất của CNN ResNet-50 và Transformer ViT-B/16 khác nhau ở đâu mà phần data augmentation của Transformer lại ít hơn CNN? Ở phần Data normalization, đoạn công thức `x_norm = (x - mean) / std` thực chất có ý nghĩa gì?

### A2.

### 1. Trước hết cần phân biệt train path và eval path

Trong notebook hiện tại:

- **train path của ResNet-50**: `Resize((256, 256)) -> RandomResizedCrop(224) -> ... -> ToTensor() -> Normalize(...)`
- **train path của ViT-B/16**: `Resize((256, 256)) -> RandomResizedCrop(224) -> ... -> ToTensor() -> Normalize(...)`
- **validation/test path dùng chung**: `Resize((256, 256)) -> CenterCrop(224) -> ToTensor() -> Normalize(...)`

Vì vậy, chuỗi `Resize((256, 256)) -> CenterCrop(224) -> ToTensor()` mà bạn hỏi chủ yếu là **đường preprocessing deterministic cho validation/test**, không phải augmentation path của training.

### 2. Tại sao lại `Resize((256, 256))` trước?

Mục tiêu chính là đưa mọi ảnh về một kích thước chuẩn trước khi crop và batching.

Lý do:

- ảnh Stanford Dogs có kích thước rất khác nhau
- CNN và ViT đều cần input có shape cố định để xếp batch
- khi mọi ảnh đã về cùng canvas `256 x 256`, bước crop sau đó trở nên đơn giản và nhất quán

Trong notebook của bạn, `Resize((256, 256))` là resize thẳng về hình vuông. Nghĩa là:

- mọi ảnh đều bị ép về `256 x 256`
- aspect ratio gốc không còn được giữ nguyên tuyệt đối
- đổi lại, pipeline đơn giản, ổn định, dễ kiểm soát và phù hợp cho benchmark thống nhất

Điểm cần hiểu kỹ:

- đây **không hoàn toàn giống** pattern ImageNet cổ điển kiểu “resize cạnh ngắn về 256 rồi center crop 224”
- notebook của bạn dùng cách thực dụng hơn: chuẩn hóa tất cả ảnh về cùng kích thước vuông trước

### 3. Tại sao sau đó lại `CenterCrop(224)`?

Sau khi ảnh đã ở `256 x 256`, crop tiếp về `224 x 224` có mấy tác dụng:

- tạo đúng input size mà backbone mong đợi
- loại bớt phần biên ngoài ảnh
- giữ phần trung tâm ổn định cho validation/test
- giúp so sánh model công bằng hơn vì evaluation không có randomness

Tại sao là `224`?

- vì cả ResNet-50 và ViT-B/16 trong notebook đều đang dùng checkpoint pretrained theo chuẩn ImageNet ở độ phân giải đầu vào `224 x 224`

Tại sao là **center** crop?

- vì ở validation/test ta muốn preprocessing lặp lại giống nhau cho mọi lần chạy
- `CenterCrop` là cách deterministic, không tạo nhiễu do random crop

Nói đơn giản:

- `256` là “khung chuẩn hóa trước”
- `224` là “kích thước input cuối cùng cho model”

### 4. Tại sao phải `ToTensor()`?

Trước `ToTensor()`, ảnh vẫn là ảnh kiểu PIL hoặc mảng pixel thông thường.

Sau `ToTensor()`:

- dữ liệu được đổi sang tensor PyTorch
- thứ tự chiều đổi từ `H x W x C` sang `C x H x W`
- giá trị pixel từ khoảng `0..255` được scale về `0..1`

Ví dụ:

- trước: ảnh RGB `224 x 224 x 3`
- sau: tensor `3 x 224 x 224`

Đây là bước bắt buộc trước khi:

- chuẩn hóa bằng `Normalize(...)`
- đưa dữ liệu vào model
- gộp thành batch `(N, C, H, W)`

### 5. Tại sao augmentation của ViT lại nhẹ hơn CNN?

Điểm này đến từ sự khác nhau trong cách hai họ backbone xử lý thông tin.

#### ResNet-50 (CNN)

CNN học đặc trưng theo kiểu cục bộ và phân cấp:

- đầu tiên học cạnh, góc, texture
- sau đó mới ghép thành bộ phận và object-level pattern
- có tính locality mạnh nhờ convolution

Vì vậy CNN thường chịu được augmentation không gian mạnh hơn một chút:

- crop mạnh hơn
- rotation
- jitter màu mạnh hơn
- erasing mạnh hơn

Trong notebook:

- ResNet crop mạnh hơn: `scale=(0.72, 1.0)`
- có thêm `RandomRotation(15)`
- `ColorJitter` mạnh hơn
- `RandomErasing(p=0.10)` cũng nhỉnh hơn

#### ViT-B/16 (Transformer)

ViT chia ảnh thành patch và học quan hệ toàn cục giữa các patch qua self-attention.

Điều đó có nghĩa:

- model rất mạnh trong việc nắm quan hệ toàn cục
- nhưng cũng nhạy hơn với thay đổi hình học quá mạnh ở bài toán fine-grained
- nếu crop quá gắt hoặc augmentation quá nặng, các chi tiết breed nhỏ như tai, mõm, texture lông có thể bị phá mất

Với Stanford Dogs là bài toán **fine-grained classification**, các dấu hiệu phân biệt rất nhỏ rất quan trọng. Vì vậy notebook chọn augmentation nhẹ hơn cho ViT:

- crop hẹp hơn: `scale=(0.78, 1.0)`
- không dùng rotation mạnh như ResNet
- `ColorJitter` nhẹ hơn
- `RandomErasing(p=0.08)` cũng nhẹ hơn

Nói ngắn gọn:

- CNN cần robustness với biến thiên cục bộ và thường chịu augmentation mạnh khá tốt
- ViT trên bài toán fine-grained dễ bị mất tín hiệu phân biệt nếu augmentation quá tay
- nên notebook cố ý dùng policy nhẹ hơn cho ViT để giữ breed cues

### 6. Phần Data normalization thực chất đang làm gì?

Công thức:

`x_norm = (x - mean) / std`

được áp dụng **theo từng channel RGB**.

Nghĩa là:

- lấy giá trị pixel của kênh đỏ, trừ mean đỏ, rồi chia cho std đỏ
- làm tương tự cho xanh lá và xanh dương

Ví dụ với kênh đỏ:

`x_red_norm = (x_red - 0.485) / 0.229`

### 7. Ý nghĩa trực giác của normalization

Normalization giúp dữ liệu đầu vào có thang đo “gọn” và ổn định hơn.

Nếu không normalize:

- các kênh màu có thể lệch scale
- gradient dễ dao động hơn
- pretrained model nhận input không giống lúc nó được huấn luyện ban đầu

Khi normalize:

- mean của mỗi channel được kéo về quanh 0
- độ phân tán được chuẩn hóa theo std
- optimization ổn định hơn
- đầu vào khớp hơn với kỳ vọng của checkpoint pretrained

### 8. Tại sao notebook dùng ImageNet mean/std thay vì mean/std đúng của Stanford Dogs?

Notebook dùng:

- `IMAGENET_MEAN = (0.485, 0.456, 0.406)`
- `IMAGENET_STD = (0.229, 0.224, 0.225)`

Lý do chính:

- cả ResNet-50 và ViT-B/16 đều khởi tạo từ checkpoint pretrained trên ImageNet
- các backbone đó đã quen với phân phối đầu vào được normalize theo ImageNet
- khi fine-tune, thường ưu tiên “đúng kỳ vọng pretrained model” hơn là “khớp tuyệt đối với dataset hiện tại”

Vì vậy, dù `metadata_with_quality.csv` cho thấy RGB mean của Stanford Dogs xấp xỉ `(0.4761, 0.4518, 0.3910)`, notebook vẫn dùng ImageNet stats để giữ compatibility với pretrained weights.

### 9. Phần custom mean/std estimation trong hình dùng để làm gì?

Đoạn code đó là mẫu để **tự tính mean và std của dataset**.

Nó làm như sau:

- cộng tổng pixel của từng channel vào `channel_sum`
- cộng tổng bình phương pixel của từng channel vào `channel_sq_sum`
- đếm tổng số pixel qua `pixel_count`
- sau đó tính:
  - `mean = channel_sum / pixel_count`
  - `std = sqrt(channel_sq_sum / pixel_count - mean^2)`

Ý nghĩa:

- nếu bạn muốn normalize theo đúng dataset của mình thì phải ước lượng mean/std như vậy
- cách này đặc biệt hữu ích khi:
  - train from scratch
  - dùng checkpoint không thuộc ImageNet family
  - dataset rất khác ImageNet

### 10. Tóm tắt ngắn

- `Resize((256, 256))` chuẩn hóa kích thước ảnh ban đầu
- `CenterCrop(224)` tạo input cuối cùng đúng chuẩn cho model và giữ evaluation ổn định
- `ToTensor()` biến ảnh thành tensor PyTorch `C x H x W` trong khoảng `0..1`
- ResNet-50 dùng augmentation mạnh hơn vì CNN thường chịu biến thiên cục bộ tốt hơn
- ViT-B/16 dùng augmentation nhẹ hơn để không phá các dấu hiệu fine-grained giữa các breed
- normalization là bước đưa từng channel về thang đo chuẩn để training ổn định và khớp với pretrained checkpoint

## Q3. `Notebook output confirms identical loader lengths for both families: 319 train, 57 val, 269 test` nghĩa là gì?

### A3.

Ba con số `319 train, 57 val, 269 test` là **số batch của DataLoader**, không phải số ảnh.

Trong notebook:

- `train` có `10,200` ảnh
- `val` có `1,800` ảnh
- `test` có `8,580` ảnh
- `BATCH_SIZE = 32`

Khi DataLoader chia dữ liệu thành từng mini-batch, số batch được tính xấp xỉ như sau:

- train: `10,200 / 32 = 318.75` nên cần `319` batch
- val: `1,800 / 32 = 56.25` nên cần `57` batch
- test: `8,580 / 32 = 268.125` nên cần `269` batch

Điều này xảy ra vì batch cuối cùng thường không đủ 32 ảnh, nhưng vẫn được giữ lại. Ví dụ:

- train có 318 batch đầy đủ và 1 batch cuối nhỏ hơn
- val có 56 batch đầy đủ và 1 batch cuối nhỏ hơn
- test có 268 batch đầy đủ và 1 batch cuối nhỏ hơn

### Vì sao cả ResNet-50 và ViT-B/16 lại có cùng các con số này?

Vì hai family dùng:

- cùng một dataset
- cùng một `split_metadata.csv`
- cùng `BATCH_SIZE = 32`
- cùng cách build `train / val / test` loaders

Khác biệt giữa hai family nằm ở **transform của train path** và **backbone model**, chứ không nằm ở số lượng mẫu hay cách chia split. Do đó:

- số ảnh trong mỗi split là như nhau
- số batch của mỗi loader cũng như nhau

Nói cách khác:

- ResNet train loader có `319` batch
- ViT train loader cũng có `319` batch

không có nghĩa là hai model giống nhau, mà chỉ có nghĩa là chúng đang được so sánh trên **cùng một benchmark setup**.

### Ý nghĩa thực tế của các con số này

Các số `319 / 57 / 269` giúp bạn hiểu:

- mỗi epoch train sẽ có `319` bước lặp gradient update
- mỗi lần validation sẽ đi qua `57` batch
- mỗi lần test/evaluation sẽ đi qua `269` batch

Ví dụ, nếu một model train `12 epochs`, thì riêng phần train loop sẽ chạy khoảng:

- `319 x 12 = 3,828` bước train batch

Con số này cũng giúp ước lượng:

- thời gian train
- số lần optimizer update
- số lần logging theo epoch hoặc theo batch

### Tóm tắt ngắn

- `319 / 57 / 269` là **số batch của DataLoader**
- không phải số ảnh
- chúng được suy ra từ số ảnh trong mỗi split chia cho `batch_size = 32`
- cả ResNet-50 và ViT-B/16 giống nhau ở chỗ này vì dùng cùng split và cùng batch size
