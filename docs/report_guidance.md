# Tài liệu Tham khảo & Cấu trúc Báo cáo Khoa học (RL Traffic Control)

Tài liệu này đề xuất các bài báo khoa học (Papers) sát nhất với dự án của bạn để tham khảo về văn phong, cấu trúc và cách trình bày kết quả.

## 1. Các Papers Đề xuất (Top-tier Conferences)

Những bài báo này được chọn vì sự tương đồng về mặt kỹ thuật (Grid network, RL, SUMO) và tiêu chuẩn chất lượng cao (KDD, AAAI, CIKM).

### 1.1 PressLight: Learning Max Pressure Control (KDD 2019)
*   **Tương đồng dự án**:
    *   Sử dụng khái niệm "Max Pressure" (liên quan đến hàm thưởng của bạn về hàng đợi/ùn tắc).
    *   Mạng lưới Grid.
    *   State representation đơn giản nhưng hiệu quả.
*   **Điểm cần học hỏi**: Cách họ biện luận (justify) cho việc thiết kế hàm thưởng (Reward Shaping) dựa trên lý thuyết giao thông (Max Pressure) thay vì chỉ dùng "thời gian chờ" một cách cảm tính.

### 1.2 MPLight: Learning Network-level Cooperation (AAAI 2020)
*   **Tương đồng dự án**:
    *   Kế thừa PressLight nhưng mở rộng cho quy mô lớn hơn.
    *   Sử dụng chia sẻ tham số (Parameter Sharing) - giống dự án của bạn (10 workers, shared model).
*   **Điểm cần học hỏi**: Cách trình bày về **Tính công bằng (Fairness)** và khả năng chuyển giao (Transferability/Generalization) trên các mạng lưới khác nhau (Unseen datasets).

### 1.3 CoLight: Learning Network-level Cooperation (CIKM 2019)
*   **Tương đồng dự án**:
    *   Tập trung vào Multi-Agent Cooperation (Hợp tác đa tác tử).
    *   Dự án của bạn dùng "Global Broadcast" để phối hợp -> CoLight dùng Graph Attention.
*   **Điểm cần học hỏi**: Cách định nghĩa vấn đề **Observability** (Khả năng quan sát) và tại sao cần giao tiếp/hợp tác giữa các giao lộ (lập luận để bảo vệ thiết kế Global State 14D của bạn).

---

## 2. Cấu trúc Báo cáo Chuẩn (Mapping với Dự án của Bạn)

Dưới đây là cấu trúc tiêu chuẩn của một paper RL-TSC và cách map nội dung dự án của bạn vào đó.

### 2.1 Abstract (Tóm tắt)
*   **Nội dung**: Vấn đề (Tắc nghẽn) -> Hạn chế giải pháp cũ (Fixed-time không linh hoạt, RL cũ khó hội tụ/không ổn định) -> Giải pháp của bạn (MARL với SMDP + Global State) -> Kết quả (Vượt trội X% so với baseline).

### 2.2 Introduction (Giới thiệu)
*   Nêu bật tính thách thức của bài toán 9 giao lộ (phối hợp).
*   Nhấn mạnh đóng góp chính của bạn:
    1.  Công thức hóa **SMDP** chuẩn (Time-aware gamma).
    2.  Kỹ thuật **Global State Broadcasting** để khôi phục tính chất Markov.
    3.  Quy trình huấn luyện **Curriculum** 9 pha để tránh quên (Catastrophic Forgetting).

### 2.3 Problem Definition (Định nghĩa Bài toán)
*   *Phần này tương ứng với mục 2 trong tài liệu Addendum của bạn.*
*   Mô tả môi trường SUMO 3x3.
*   Trình bày hình thức hóa Markov Game (States, Actions, Rewards).
*   **Mẹo**: Sử dụng các công thức toán học cho Reward (như bạn đã làm trong Addendum) nhìn rất chuyên nghiệp.

### 2.4 Methodology (Phương pháp)
*   **Agent Design**: Mô tả kiến trúc Double Dueling DQN.
*   **Cooperation Mechanism**: Giải thích tại sao thêm 2 dimensions (Global N & Spillback) vào State. Đây là điểm mấu chốt (Novelty).
*   **Training Strategy**: Mô tả Parallel Training (10 workers) và Curriculum Learning. Vẽ sơ đồ luồng (Workflow diagram) ở đây.

### 2.5 Experiments (Thực nghiệm)
*   **Experimental Setup**:
    *   Datasets: Mô tả các mức demand (500, 750, 1000).
    *   Baselines: Fixed-time, Max-Pressure (quan trọng nhất để so sánh).
*   **Evaluation Metrics**: Waiting time, Queue length, Throughput.

### 2.6 Results & Analysis (Kết quả & Phân tích)
*   **Overall Performance**: Biểu đồ so sánh KPI với baselines.
*   **Ablation Study (Quan trọng)**: Phần này bạn đã có data (hoặc plan có data).
    *   *RL vs RL no-global-state*: Chứng minh Global State có tác dụng.
    *   *SMDP vs User-defined Reward*: Chứng minh công thức thưởng SMDP tốt hơn.
*   **Generalization**: Kết quả chạy trên Demand 1250 (Unseen).

### 2.7 Conclusion (Kết luận)
*   Tóm tắt lại thành quả.
*   Hướng phát triển (Future work): Mạng lưới lớn hơn, dữ liệu thực tế.

---

## 3. Lời khuyên cho "Report Writing"
1.  **"Don't just show, Explain"**: Đừng chỉ đưa ra biểu đồ. Hãy giải thích *tại sao* đường biểu đồ lại đi như vậy (VD: Tại sao đoạn đầu reward dao động mạnh? Do Epsilon cao).
2.  **So sánh công bằng**: Nhấn mạnh rằng bạn so sánh với Max-Pressure trên cùng một Action Space (Discrete) để đảm bảo tính công bằng.
3.  **Visualization**:
    *   Dùng sơ đồ (Diagram) cho kiến trúc mạng và luồng huấn luyện.
    *   Dùng bảng (Table) thay vì text để liệt kê tham số (như Addendum bạn đã làm rất tốt).

Bạn có thể tìm đọc file PDF của các paper trên (thường có bản free trên arXiv) để bắt chước cách họ viết các caption cho hình ảnh và bảng biểu.
