![z6917654370462_2817f81f026235364ef69cc6d6056fcc](https://github.com/user-attachments/assets/5d618a40-9dc5-4d9c-954e-0430c5f00c9d)
Hệ thống nhận dạng & Phân loại giấy tờ bằng AI
Giới thiệu
Dự án này xây dựng hệ thống nhận dạng và phân loại giấy tờ (ứng minh thư, hộ chiếu, giấy phép lái xe, cấp, hóa đơn, vv) dựa trên:

OCR (Nhận dạng ký tự quang học) : trích xuất văn bản từ ảnh giấy tờ bằng PaddleOCR và Tesseract .
PhoBERT : mô hình ngôn ngữ tiếng Việt để sinh vector đặc trưng cho văn bản.
Hồi quy logistic : phân loại văn bản giấy tờ dựa trên nhúng PhoBERT.
Streamlit : giao diện web trực quan để người dùng tải ảnh, xem kết quả OCR và loại giấy tờ dự kiến.
Kiến trúc hệ thống
Tiền xử lý ảnh : Chuyển xám, làm mờ, nhị phân, làm sắc nét bằng OpenCV.
OCR : Kết hợp PaddleOCR (chính) và Tesseract (bổ sung).
Embedding : Văn bản OCR được mã hóa thành vector 768 chiều bởi PhoBERT.
Phân loại : Logistic Regression dự kiến ​​loại giấy tờ.
Giao diện : Streamlit hiển thị hình ảnh, văn bản OCR và phân loại kết quả.
Cấu trúc thư mục
.
├── app.py               # Ứng dụng Streamlit (demo)
├── train.py             # Script huấn luyện mô hình phân loại
├── dataset/             # Dữ liệu huấn luyện (ảnh chia theo thư mục nhãn)
│   ├── cccd/
│   ├── ho_chieu/
│   └── bang_lai/
├── doc_classifier.pkl   # Mô hình Logistic Regression đã huấn luyện
├── labels.json          # Danh sách nhãn giấy tờ
├── requirements.txt     # Thư viện cần cài
Cài đặt môi trường
1. Sao chép repo & tạo môi trường
git clone <repo_url>
cd <repo_name>
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
2. Cài đặt thư viện
pip install -r requirements.txt
Nội dung requirements.txtgợi ý:

streamlit
opencv-python
numpy
pytesseract
Pillow
torch
transformers
scikit-learn
joblib
paddleocr
3. Cài đặt Tesseract
Windows : Tải xuống Tesseract OCR
Linux/Mac :
sudo apt install tesseract-ocr
Huấn luyện mô hình
Chuẩn bị dữ liệu theo cấu trúc:

dataset/
├── cccd/        # ảnh Căn cước công dân
├── ho_chieu/    # ảnh Hộ chiếu
└── bang_lai/    # ảnh Bằng lái xe
Chạy tập lệnh đào tạo:

python train.py
Kết quả:
doc_classifier.pkl: mô hình Logistic Regression đã được huấn luyện.
labels.json: danh sách nhãn tương ứng.
Chạy ứng dụng
streamlit run app.py
Tải ảnh giấy tờ ( .jpg, .jpeg, .png).
Hệ thống sẽ:
Tiền xử lý ảnh.
Nhận văn bản dạng bằng OCR.
Nhúng vector sinh học bằng PhoBERT.
Dự kiến ​​loại giấy tờ và hiển thị kết quả.
Ví dụ kết quả
Ảnh gốc & tiền xử lý : hiển thị bài hát.
Văn bản OCR : nội dung trích xuất từ ​​ảnh.
Loại giấy tờ dự kiến ​​: Ví dụ: Căn chân công dân .
Hướng phát triển
Bổ sung thêm nhiều loại giấy tờ (bằng cấp, hóa đơn, hợp đồng...).
Tinh chỉnh PhoBERT để tăng độ chính xác.
Tích hợp API REST để phát triển khai thực tế.
Hỗ trợ nhiều ngôn ngữ ngoài tiếng Việt.
