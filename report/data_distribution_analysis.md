# Phân tích Định lượng Dữ liệu Thực nghiệm (Dataset MNIST)

Báo cáo này phân tích chi tiết hai khía cạnh chính của dữ liệu được sử dụng trong các vòng lặp Federated Learning:
1. **Phân phối nhãn (Label Distribution):** Các kịch bản K1, K2, K3 ảnh hưởng thế nào đến sự phân bổ class trên từng client.
2. **Phân phối quy mô dữ liệu (Data Size Imbalance):** Các kịch bản mất cân bằng số lượng mẫu dữ liệu ảnh hưởng thế nào đến số lượng data của từng client.

---

## 1. Phân phối Nhãn dữ liệu (Label Distribution)
Dưới đây là thống kê chi tiết số lượng mẫu và phân bố class (3 class xuất hiện nhiều nhất) của **TOÀN BỘ 10 CLIENT**, sử dụng dataset MNIST (60,000 mẫu train, 10 class).

| Kịch bản | Client ID | Tổng mẫu | Số class | Phân bố mẫu theo Class (Top 3) |
|---|---|---|---|---|
| **K1 (IID)** | Client 0 | 6000 | 10 | Class 1: 675, Class 3: 644, Class 8: 608 |
| **K1 (IID)** | Client 1 | 6000 | 10 | Class 1: 694, Class 3: 623, Class 7: 614 |
| **K1 (IID)** | Client 2 | 6000 | 10 | Class 1: 673, Class 3: 651, Class 7: 615 |
| **K1 (IID)** | Client 3 | 6000 | 10 | Class 1: 658, Class 7: 604, Class 2: 600 |
| **K1 (IID)** | Client 4 | 6000 | 10 | Class 1: 687, Class 8: 635, Class 6: 609 |
| **K1 (IID)** | Client 5 | 6000 | 10 | Class 1: 647, Class 7: 636, Class 9: 607 |
| **K1 (IID)** | Client 6 | 6000 | 10 | Class 7: 692, Class 1: 660, Class 6: 613 |
| **K1 (IID)** | Client 7 | 6000 | 10 | Class 1: 673, Class 7: 634, Class 0: 630 |
| **K1 (IID)** | Client 8 | 6000 | 10 | Class 1: 657, Class 9: 637, Class 7: 636 |
| **K1 (IID)** | Client 9 | 6000 | 10 | Class 1: 718, Class 7: 627, Class 0: 620 |
| **K2 (Weak Non-IID)** | Client 0 | 3063 | 5 | Class 1: 675, Class 3: 614, Class 2: 596 |
| **K2 (Weak Non-IID)** | Client 1 | 2943 | 5 | Class 7: 627, Class 2: 596, Class 0: 593 |
| **K2 (Weak Non-IID)** | Client 2 | 2928 | 5 | Class 3: 613, Class 2: 596, Class 0: 593 |
| **K2 (Weak Non-IID)** | Client 3 | 2941 | 5 | Class 7: 627, Class 9: 595, Class 6: 592 |
| **K2 (Weak Non-IID)** | Client 4 | 3059 | 5 | Class 1: 674, Class 3: 613, Class 2: 596 |
| **K2 (Weak Non-IID)** | Client 5 | 3009 | 5 | Class 1: 674, Class 3: 613, Class 9: 595 |
| **K2 (Weak Non-IID)** | Client 6 | 3059 | 5 | Class 1: 674, Class 3: 613, Class 2: 596 |
| **K2 (Weak Non-IID)** | Client 7 | 2947 | 5 | Class 7: 626, Class 9: 595, Class 6: 592 |
| **K2 (Weak Non-IID)** | Client 8 | 3008 | 5 | Class 1: 674, Class 3: 613, Class 2: 595 |
| **K2 (Weak Non-IID)** | Client 9 | 2938 | 5 | Class 7: 626, Class 9: 594, Class 6: 591 |
| **K3 (Dirichlet a=0.5)** | Client 0 | 5669 | 10 | Class 1: 1510, Class 2: 1007, Class 4: 942 |
| **K3 (Dirichlet a=0.5)** | Client 1 | 6558 | 10 | Class 9: 2764, Class 7: 2528, Class 8: 512 |
| **K3 (Dirichlet a=0.5)** | Client 2 | 5193 | 10 | Class 0: 1453, Class 2: 1064, Class 7: 1052 |
| **K3 (Dirichlet a=0.5)** | Client 3 | 9858 | 10 | Class 1: 2598, Class 0: 2530, Class 7: 1574 |
| **K3 (Dirichlet a=0.5)** | Client 4 | 4116 | 10 | Class 5: 887, Class 8: 788, Class 2: 714 |
| **K3 (Dirichlet a=0.5)** | Client 5 | 4606 | 10 | Class 3: 1183, Class 8: 968, Class 9: 579 |
| **K3 (Dirichlet a=0.5)** | Client 6 | 10756 | 10 | Class 4: 3128, Class 6: 2425, Class 2: 2243 |
| **K3 (Dirichlet a=0.5)** | Client 7 | 3576 | 10 | Class 3: 1634, Class 1: 883, Class 8: 545 |
| **K3 (Dirichlet a=0.5)** | Client 8 | 5684 | 10 | Class 5: 1817, Class 9: 971, Class 8: 955 |
| **K3 (Dirichlet a=0.5)** | Client 9 | 3984 | 10 | Class 8: 1743, Class 9: 777, Class 1: 677 |
| **K3 (Dirichlet a=0.1)** | Client 0 | 3187 | 9 | Class 8: 1771, Class 2: 573, Class 6: 443 |
| **K3 (Dirichlet a=0.1)** | Client 1 | 5858 | 6 | Class 7: 4225, Class 1: 801, Class 9: 771 |
| **K3 (Dirichlet a=0.1)** | Client 2 | 4156 | 8 | Class 7: 1863, Class 9: 1510, Class 2: 684 |
| **K3 (Dirichlet a=0.1)** | Client 3 | 11604 | 8 | Class 3: 5522, Class 9: 3572, Class 0: 1619 |
| **K3 (Dirichlet a=0.1)** | Client 4 | 7580 | 7 | Class 0: 3872, Class 5: 3481, Class 2: 166 |
| **K3 (Dirichlet a=0.1)** | Client 5 | 1093 | 4 | Class 5: 953, Class 1: 120, Class 4: 16 |
| **K3 (Dirichlet a=0.1)** | Client 6 | 11168 | 7 | Class 1: 5126, Class 6: 5071, Class 5: 577 |
| **K3 (Dirichlet a=0.1)** | Client 7 | 7749 | 6 | Class 2: 4468, Class 8: 3118, Class 0: 79 |
| **K3 (Dirichlet a=0.1)** | Client 8 | 937 | 6 | Class 3: 377, Class 0: 265, Class 4: 145 |
| **K3 (Dirichlet a=0.1)** | Client 9 | 6668 | 9 | Class 4: 5503, Class 8: 945, Class 5: 122 |

> [!NOTE] 
> **Nhận xét:**
> - K1 (IID): Dữ liệu cực kỳ đồng đều trên toàn bộ 10 client, mỗi client có đúng 6000 mẫu và phân bố đều cho cả 10 class.
> - K2 (Weak Non-IID): Mỗi client bị giới hạn nghiêm ngặt chỉ sở hữu đúng 5 class. Các client có khoảng 2900 - 3000 mẫu.
> - K3 (Dirichlet): Khi $\alpha$ càng nhỏ (0.1 so với 0.5), độ lệch chuẩn càng khắc nghiệt. Những client như Client 5 ở $\alpha=0.1$ chỉ có vỏn vẹn 1093 mẫu phân bố trên 4 class, trong khi Client 3 lại ôm tới 11604 mẫu. Sự chênh lệch dữ dội này là đặc điểm cốt lõi của bài toán phân phối Dirichlet.

---

## 2. Phân phối Quy mô Dữ liệu (Data Size Imbalance)
Thống kê số lượng mẫu của toàn bộ 10 client dưới các cơ chế tạo độ lệch quy mô dữ liệu (được downsample từ phân phối K1 ban đầu). 

| Kịch bản Imbalance | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | Max / Min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Uniform** (Cơ bản) | 6000 | 6000 | 6000 | 6000 | 6000 | 6000 | 6000 | 6000 | 6000 | 6000 | **1.0x** |
| **Step** (Phân cực) | 6000 | 2000 | 6000 | 6000 | 6000 | 6000 | 2000 | 2000 | 2000 | 2000 | **3.0x** |
| **Linear** (Tuyến tính) | 4000 | 2500 | 6000 | 5500 | 5000 | 4500 | 1500 | 3500 | 3000 | 2000 | **4.0x** |
| **Lognormal** (Thực tế) | 1729 | 6000 | 4643 | 400 | 2116 | 3488 | 1287 | 957 | 3691 | 3895 | **15.0x** |

> [!TIP]
> **Vai trò của Data Size Imbalance trong Đánh giá CSRA:**
> - Nếu sử dụng kịch bản Uniform, mọi client đều có khối lượng data giống nhau, vô tình làm cho chính sách phần thưởng DataSize Policy mất đi ý nghĩa đánh giá.
> - Kịch bản **Lognormal** tạo ra môi trường tiệm cận thực tế nhất: có những client khổng lồ (Client 1: 6000 mẫu) và những client có thiết bị cực nhỏ (Client 3: 400 mẫu). Chênh lệch lên đến 15 lần. 
> - Điều này đặc biệt quan trọng để test sự công bằng của chính sách **CSRAReward**, khi phải dung hòa giữa những client đóng góp dữ liệu khổng lồ nhưng chất lượng trung bình, và những client ít dữ liệu nhưng chất lượng rất cao.


---

## 3. Phân phối Nhãn dữ liệu (Dataset Fashion-MNIST)
Dưới đây là thống kê chi tiết cho **Fashion-MNIST** (60,000 mẫu train, 10 class). Quá trình phân chia chia sẻ cùng thuật toán và seed=42, nhưng do đặc thù thứ tự dữ liệu gốc, phân bố nhãn sẽ có đôi chút khác biệt so với MNIST.

| Kịch bản | Client ID | Tổng mẫu | Số class | Phân bố mẫu theo Class (Top 3) |
|---|---|---|---|---|
| **K1 (IID)** | Client 0 | 6000 | 10 | Class 9: 624, Class 1: 614, Class 2: 610 |
| **K1 (IID)** | Client 1 | 6000 | 10 | Class 9: 640, Class 6: 627, Class 5: 622 |
| **K1 (IID)** | Client 2 | 6000 | 10 | Class 0: 623, Class 5: 621, Class 7: 617 |
| **K1 (IID)** | Client 3 | 6000 | 10 | Class 2: 639, Class 0: 636, Class 1: 623 |
| **K1 (IID)** | Client 4 | 6000 | 10 | Class 1: 623, Class 6: 623, Class 9: 618 |
| **K1 (IID)** | Client 5 | 6000 | 10 | Class 6: 651, Class 3: 639, Class 2: 624 |
| **K1 (IID)** | Client 6 | 6000 | 10 | Class 8: 646, Class 1: 636, Class 0: 615 |
| **K1 (IID)** | Client 7 | 6000 | 10 | Class 4: 655, Class 6: 610, Class 0: 610 |
| **K1 (IID)** | Client 8 | 6000 | 10 | Class 4: 641, Class 8: 610, Class 5: 609 |
| **K1 (IID)** | Client 9 | 6000 | 10 | Class 3: 649, Class 6: 622, Class 7: 613 |
| **K2 (Weak Non-IID)** | Client 0 | 3000 | 5 | Class 0: 600, Class 1: 600, Class 2: 600 |
| **K2 (Weak Non-IID)** | Client 1 | 3000 | 5 | Class 6: 600, Class 7: 600, Class 9: 600 |
| **K2 (Weak Non-IID)** | Client 2 | 3000 | 5 | Class 0: 600, Class 3: 600, Class 2: 600 |
| **K2 (Weak Non-IID)** | Client 3 | 3000 | 5 | Class 5: 600, Class 6: 600, Class 7: 600 |
| **K2 (Weak Non-IID)** | Client 4 | 3000 | 5 | Class 1: 600, Class 2: 600, Class 4: 600 |
| **K2 (Weak Non-IID)** | Client 5 | 3000 | 5 | Class 7: 600, Class 8: 600, Class 0: 600 |
| **K2 (Weak Non-IID)** | Client 6 | 3000 | 5 | Class 0: 600, Class 1: 600, Class 2: 600 |
| **K2 (Weak Non-IID)** | Client 7 | 3000 | 5 | Class 5: 600, Class 6: 600, Class 8: 600 |
| **K2 (Weak Non-IID)** | Client 8 | 3000 | 5 | Class 0: 600, Class 3: 600, Class 2: 600 |
| **K2 (Weak Non-IID)** | Client 9 | 3000 | 5 | Class 5: 600, Class 6: 600, Class 7: 600 |
| **K3 (Dirichlet a=0.5)** | Client 0 | 11815 | 10 | Class 3: 2363, Class 6: 2105, Class 9: 1685 |
| **K3 (Dirichlet a=0.5)** | Client 1 | 1816 | 10 | Class 1: 705, Class 6: 565, Class 3: 245 |
| **K3 (Dirichlet a=0.5)** | Client 2 | 4016 | 9 | Class 8: 1652, Class 6: 1348, Class 7: 501 |
| **K3 (Dirichlet a=0.5)** | Client 3 | 6621 | 9 | Class 1: 2698, Class 9: 1580, Class 7: 1042 |
| **K3 (Dirichlet a=0.5)** | Client 4 | 9072 | 10 | Class 3: 2406, Class 5: 2299, Class 9: 1301 |
| **K3 (Dirichlet a=0.5)** | Client 5 | 10960 | 10 | Class 7: 2932, Class 2: 2762, Class 4: 1165 |
| **K3 (Dirichlet a=0.5)** | Client 6 | 2545 | 10 | Class 5: 816, Class 4: 629, Class 1: 402 |
| **K3 (Dirichlet a=0.5)** | Client 7 | 3244 | 10 | Class 4: 1096, Class 7: 679, Class 2: 431 |
| **K3 (Dirichlet a=0.5)** | Client 8 | 3669 | 10 | Class 0: 2063, Class 5: 467, Class 2: 311 |
| **K3 (Dirichlet a=0.5)** | Client 9 | 6242 | 10 | Class 8: 2239, Class 2: 1385, Class 6: 851 |
| **K3 (Dirichlet a=0.1)** | Client 0 | 6541 | 8 | Class 3: 2857, Class 4: 2409, Class 8: 985 |
| **K3 (Dirichlet a=0.1)** | Client 1 | 13622 | 7 | Class 0: 4646, Class 1: 4279, Class 8: 4109 |
| **K3 (Dirichlet a=0.1)** | Client 2 | 4625 | 4 | Class 6: 4423, Class 1: 200, Class 4: 1 |
| **K3 (Dirichlet a=0.1)** | Client 3 | 1448 | 6 | Class 7: 919, Class 4: 197, Class 0: 175 |
| **K3 (Dirichlet a=0.1)** | Client 4 | 8584 | 7 | Class 5: 3853, Class 3: 2949, Class 6: 1012 |
| **K3 (Dirichlet a=0.1)** | Client 5 | 9972 | 7 | Class 2: 5536, Class 4: 1472, Class 1: 1438 |
| **K3 (Dirichlet a=0.1)** | Client 6 | 2439 | 5 | Class 8: 791, Class 0: 690, Class 5: 639 |
| **K3 (Dirichlet a=0.1)** | Client 7 | 2011 | 8 | Class 4: 1297, Class 0: 484, Class 9: 206 |
| **K3 (Dirichlet a=0.1)** | Client 8 | 5875 | 6 | Class 9: 5781, Class 5: 64, Class 2: 20 |
| **K3 (Dirichlet a=0.1)** | Client 9 | 4883 | 10 | Class 7: 4734, Class 5: 135, Class 2: 4 |
