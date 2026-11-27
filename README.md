# nmpc_m0609

day la ban mo phong bo dieu khien NMPC cho robot **Doosan M0609** gắn module **Doosan SVM**, tay kẹp **Robotiq 2F-85**, camera **RealSense D435**, mô phỏng trên **ROS 2 Jazzy + Gazebo Harmonic**.

---

## 1. Tính năng chính
- **Bộ điều khiển NMPC (`acados` + C++ 50 Hz):** Tránh va chạm toàn thân (`elbow → ee`), tránh va chạm kể cả khi robot đang đứng yên tại `HOME`, quỹ đạo vòng cung vượt vật cản trơn tru.
- **2 Kịch bản Gazebo:**
  - **Bản 1 — Thanh Ray (Mặc định):** Bàn gỗ `110 × 40 cm`, băng tải cấp phôi có khung ranh giới Ô gắp, thanh ray `40 cm` xoay `90°` mang trụ cản Vàng (`Ø10 × 26 cm`), lưới xếp `2×2` và vùng trống Manual.
  - **Bản 2 — Vật Động:** Trụ Vàng di chuyển quỹ đạo **Tròn** hoặc **Zigzag** quanh vùng trống `30 cm` giữa băng tải và `HOME`.
- **Vision & Tab 3 (`3. Vision Pick`):**
  - Nhận diện đồng thời **Phôi Vuông Cam (`6×6×6 cm`)**, **Phôi Trụ Xanh (`Ø6×6 cm`)**, và **Trụ Cản Vàng (`Ø10×26 cm`)**.
  - Khóa vùng gắp theo đúng ranh giới Ô gắp trên băng tải (`y ∈ [-0.495, -0.295] m`), có nút **Chạy Băng Tải** đưa phôi vào ô.
  - Hỗ trợ 2 chế độ: **Auto Liên Tục (Xếp Ô 2×2)** và **Manual 1 Lần (Vùng Trống)**.

---

## 2. Cài đặt tự động trên máy mới (Ubuntu 24.04 + ROS 2 Jazzy)

Trên một máy tính hoàn toàn mới, chỉ cần clone repo này vào workspace bất kỳ (ví dụ `~/doosan_ws/src`) và chạy file `setup_environment.sh` — script sẽ tự cài ROS 2/Gazebo/OpenCV/PyQt5, tải và biên dịch `acados`, clone `doosan-robot2`, build lại solver NMPC và `colcon build`:

```bash
mkdir -p ~/doosan_ws/src
cd ~/doosan_ws/src
git clone https://github.com/minhductcvt/nmpc_m0609.git dsr_m0609_nmpc

cd dsr_m0609_nmpc
chmod +x setup_environment.sh
./setup_environment.sh
```

> *(Mở xem sơ đồ thiết kế trạm máy offline bằng trình duyệt: mở file `docs/cell_layout_blueprint.html`)*

---

## 3. Hướng dẫn khởi động & thao tác GUI

### Khởi động giao diện Teach Pendant
Mở Terminal tại thư mục workspace (ví dụ `~/doosan_ws`) và chạy:
```bash
cd ~/doosan_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch dsr_m0609_nmpc teach_pendant.launch.py
```

### Quy trình vận hành chuẩn trên GUI (6 Tab)

1. **Khởi động Gazebo & Bộ điều khiển NMPC (`Tab 6. Process`):**
   - Sang tab **`6. Process`**.
   - Bấm **`▶ Bật Gazebo`** (mặc định mở bản Thanh Ray; muốn chuyển sang bản Vật Động Tròn/Zigzag thì sang tab **`2. Dynamic Lab`** để chọn).
   - Đợi cửa sổ Gazebo Sim hiện lên robot Doosan M0609 xong, bấm **`▶ Bật NMPC Solver`**.
   - *(Tuỳ chọn)* Bấm **`▶ Bật Vision Node`** ngay tại đây hoặc bật ở Tab 3.

2. **Chuyển kịch bản Vật Cản (`Tab 2. Dynamic Lab`):**
   - **Bản 1 (Thanh Ray - Mặc định):** Có thể tick **`Trượt trụ cản qua lại trên thanh ray`** để cho trụ Vàng chạy dọc ray (robot đứng ở `HOME` sẽ tự nhấc né).
   - **Bản 2 (Vật Động):** Chọn **`Vật Cản Động`**, sau đó bấm **`Quỹ Đạo Tròn`** hoặc **`Quỹ Đạo Zigzag`**, điều chỉnh tốc độ (`0.3x – 2.5x`) hoặc bấm **`⏸ Tạm Dừng / Tiếp Tục`**.

3. **Gắp & Xếp Phôi Tự Động / Thủ Công (`Tab 3. Vision Pick`):**
   - Bấm **`▶ Bật Vision`** ở góc trên khung camera D435 để hiện nhận diện 3D + khung **`O GAP VAT`** trên băng tải.
   - Kiểm tra mục **`1. Phôi Nhận Diện Trên Băng Tải`**: danh sách hiển thị đồng thời **Vuông Cam** và **Trụ Xanh** kèm trạng thái (`🟢 TRONG Ô GẮP` hoặc `🟡 Chờ ngoài ô gắp`).
   - Bấm **`🏭 Chạy Băng Tải (Đưa vào Ô gắp)`** nếu phôi vẫn đang nằm ngoài vạch ranh giới băng tải.
   - Chọn 1 trong 2 chế độ ở mục **`2. Chế Độ Gắp`**:
     - **`🤖 Auto Liên Tục (Xếp Ô 2×2)`**: Bấm **`▶ CHẠY AUTO LIÊN TỤC`** → robot tự gắp phôi trong ô gắp, bay vòng cung vượt trụ Vàng, xếp Vuông Cam vào ô Cam và Trụ Xanh vào ô Xanh của lưới `2×2` (tự đẩy phôi tiếp theo nếu tick `Tự đẩy phôi băng tải`).
     - **`🖐️ Manual 1 Lần (Vùng Trống)`**: Chọn phôi muốn gắp (Cam hoặc Xanh), dùng cụm **`Jog XYZ + R`** đưa tay máy vào vùng trống `+Y` rồi bấm **`📍 XÁC NHẬN VỊ TRÍ ĐẶT`**, sau đó bấm **`▶ CHẠY MANUAL 1 LẦN`** để thực hiện đúng 1 chu trình.
   - Bấm **`🔄 Reset Ô & Phôi`** để đặt lại 2 phôi lên băng tải và làm mới bộ đếm `2×2`.

4. **Các Tab còn lại:**
   - **`1. Control & Safety`**: Radar 2D, theo dõi khoảng cách an toàn 3D theo thời gian thực, điều khiển khớp/Jog 6-DOF và thanh trượt kẹp Robotiq 2F-85.
   - **`4. Task Builder`**: Dạy điểm và soạn chuỗi lệnh `MoveJ` / `MoveNMPC` / `Gripper` / `Wait`.
   - **`5. NMPC Setup`**: Chỉnh trực tiếp tọa độ vật cản tĩnh & ngưỡng khoảng cách an toàn.
