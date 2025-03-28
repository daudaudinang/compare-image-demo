# ui_comparator/visualization.py

"""Module hiển thị kết quả trực quan."""

import cv2
import matplotlib
import matplotlib.pyplot as plt
from ui_comparator.analysis import DifferenceAnalyzer

matplotlib.use('TkAgg')  # Sử dụng backend TkAgg thay vì Qt

# --- Định nghĩa màu sắc (BGR) ---
COLOR_GREEN = (0, 255, 0)      # Rất tương đồng (> 0.9)
COLOR_YELLOW = (0, 255, 255)   # Khớp nhưng có khác biệt (<= 0.9 và > threshold)
COLOR_RED = (0, 0, 255)        # Không khớp / Khác biệt lớn / Thiếu / Thừa
# --------------------------------

class ResultVisualizer:
    """Hiển thị kết quả trực quan."""

    def __init__(self):
        pass

    def draw_bounding_boxes(self, img, elements, color=(0, 255, 0), show_indices=True):
        """Vẽ bounding box cho các thành phần UI (hữu ích để xem segmentation riêng)."""
        result = img.copy()

        for i, element in enumerate(elements):
            try:
                # Đảm bảo bbox là tuple/list gồm 4 số nguyên
                x, y, w, h = map(int, element.get('bbox', (0,0,0,0)))
                if w > 0 and h > 0: # Chỉ vẽ nếu bbox hợp lệ
                    cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
                    if show_indices:
                        cv2.putText(result, f"UI-{i}", (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1) # Giảm độ dày
            except (ValueError, TypeError, KeyError) as e:
                 # In cảnh báo nếu có lỗi nhưng không dừng chương trình
                 print(f"Warning: Lỗi khi vẽ bbox cho element {i}: {e}, bbox: {element.get('bbox')}")

        return result

    def draw_all_elements(self, img1, img2, elements1, elements2, matches, analysis_result, match_threshold=0.55):
        """
        Vẽ tất cả các phần tử lên ảnh gốc với màu sắc tương ứng:
        - Xanh lá: Khớp tốt (> 0.90)
        - Vàng: Khớp vừa phải (> threshold và <= 0.90)
        - Đỏ: Không khớp (thiếu/thừa)

        Args:
            img1: Ảnh design gốc (numpy array BGR).
            img2: Ảnh thực tế gốc (numpy array BGR).
            elements1: List các element của ảnh design.
            elements2: List các element của ảnh thực tế.
            matches: List các cặp khớp từ ElementMatcher.
            analysis_result: Dict kết quả từ DifferenceAnalyzer (chứa 'missing_elements', 'extra_elements').
            match_threshold (float): Ngưỡng similarity tối thiểu để coi là khớp (dùng để phân biệt Vàng/Đỏ nếu cần).
        """
        vis1 = img1.copy()
        vis2 = img2.copy()

        # --- Tạo map từ id object sang index để tìm nhanh hơn ---
        # Giả định mỗi element object là duy nhất trong list của nó
        element1_to_index = {id(elem): i for i, elem in enumerate(elements1)}
        element2_to_index = {id(elem): i for i, elem in enumerate(elements2)}
        # -------------------------------------------------------

        matched_indices1 = set()
        matched_indices2 = set()

        # Vẽ các phần tử đã khớp (Xanh lá hoặc Vàng)
        for i, match in enumerate(matches):
            element1 = match.get('element1')
            element2 = match.get('element2')
            similarity = match.get('similarity', 0.0)

            if element1 is None or element2 is None:
                print(f"Warning: Match {i} thiếu element1 hoặc element2.")
                continue

            # Xác định màu dựa trên similarity
            color = COLOR_GREEN if similarity > 0.9 else COLOR_YELLOW

            # --- Vẽ bbox cho element1 trên vis1 ---
            try:
                bbox1 = element1.get('bbox')
                if bbox1:
                    x1, y1, w1, h1 = map(int, bbox1)
                    if w1 > 0 and h1 > 0:
                        cv2.rectangle(vis1, (x1, y1), (x1 + w1, y1 + h1), color, 2)
                        # Optional: Hiển thị similarity score
                        cv2.putText(vis1, f"{similarity:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                        # Lưu index của element đã được vẽ
                        idx1 = element1_to_index.get(id(element1))
                        if idx1 is not None:
                            matched_indices1.add(idx1)
                        else:
                            # Thử tìm bằng cách so sánh trực tiếp nếu id không khớp (chậm hơn)
                            try:
                                idx1_fallback = elements1.index(element1)
                                matched_indices1.add(idx_fallback)
                            except ValueError:
                                print(f"Warning: Không tìm thấy index cho element1 trong match {i} (cả bằng id và value)")

            except (ValueError, TypeError, KeyError) as e:
                print(f"Lỗi khi vẽ match element 1 (match index {i}): {e}, bbox: {element1.get('bbox')}")

            # --- Vẽ bbox cho element2 trên vis2 ---
            try:
                bbox2 = element2.get('bbox')
                if bbox2:
                    x2, y2, w2, h2 = map(int, bbox2)
                    if w2 > 0 and h2 > 0:
                        cv2.rectangle(vis2, (x2, y2), (x2 + w2, y2 + h2), color, 2)
                        # Optional: Hiển thị similarity score
                        cv2.putText(vis2, f"{similarity:.2f}", (x2, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                        # Lưu index của element đã được vẽ
                        idx2 = element2_to_index.get(id(element2))
                        if idx2 is not None:
                            matched_indices2.add(idx2)
                        else:
                             # Thử tìm bằng cách so sánh trực tiếp
                            try:
                                idx2_fallback = elements2.index(element2)
                                matched_indices2.add(idx2_fallback)
                            except ValueError:
                                print(f"Warning: Không tìm thấy index cho element2 trong match {i} (cả bằng id và value)")

            except (ValueError, TypeError, KeyError) as e:
                 print(f"Lỗi khi vẽ match element 2 (match index {i}): {e}, bbox: {element2.get('bbox')}")


        # --- Vẽ các phần tử không khớp (Đỏ) ---

        # Phần tử thiếu trong ảnh thực tế (có trong design, không khớp)
        # Lấy trực tiếp từ analysis_result để đảm bảo đúng danh sách
        missing_elements = analysis_result.get('missing_elements', [])
        print(f"Số phần tử thiếu (vẽ màu đỏ trên design): {len(missing_elements)}")
        for element in missing_elements:
             try:
                 bbox = element.get('bbox')
                 if bbox:
                     x, y, w, h = map(int, bbox)
                     if w > 0 and h > 0:
                         cv2.rectangle(vis1, (x, y), (x + w, y + h), COLOR_RED, 2)
                         # Optional: Thêm text "Missing"
                         cv2.putText(vis1, "Missing", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_RED, 1)
             except (ValueError, TypeError, KeyError) as e:
                  print(f"Lỗi khi vẽ missing element: {e}, bbox: {element.get('bbox')}")


        # Phần tử thừa trong ảnh thực tế (có trong real, không khớp)
        extra_elements = analysis_result.get('extra_elements', [])
        print(f"Số phần tử thừa (vẽ màu đỏ trên real): {len(extra_elements)}")
        for element in extra_elements:
             try:
                 bbox = element.get('bbox')
                 if bbox:
                     x, y, w, h = map(int, bbox)
                     if w > 0 and h > 0:
                         cv2.rectangle(vis2, (x, y), (x + w, y + h), COLOR_RED, 2)
                         # Optional: Thêm text "Extra"
                         cv2.putText(vis2, "Extra", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_RED, 1)
             except (ValueError, TypeError, KeyError) as e:
                  print(f"Lỗi khi vẽ extra element: {e}, bbox: {element.get('bbox')}")

        return vis1, vis2

    def create_difference_report(self, analysis_result, overall_score):
        """Tạo báo cáo về sự khác biệt."""
        # --- Code giữ nguyên, thêm .get() để an toàn hơn ---
        matched = analysis_result.get('matched_elements', [])
        missing = analysis_result.get('missing_elements', [])
        extra = analysis_result.get('extra_elements', [])

        diff_counts = {
            'Giống hệt': 0,
            'Gần giống': 0,
            'Khác biệt nhẹ': 0,
            'Khác biệt lớn': 0
        }
        for diff in matched:
            # Dùng similarity để phân loại lại nếu 'type' không có
            similarity = diff.get('similarity', 0.0)
            diff_type = diff.get('type')
            if diff_type is None: # Tự phân loại nếu analyzer không cung cấp
                if similarity > 0.9: diff_type = "Giống hệt"
                elif similarity > 0.7: diff_type = "Gần giống"
                elif similarity > 0.5: diff_type = "Khác biệt nhẹ" # Giả sử ngưỡng match là 0.5
                else: diff_type = "Khác biệt lớn"

            if diff_type in diff_counts:
                 diff_counts[diff_type] += 1
            else: # Xử lý type lạ nếu có
                 diff_counts['Khác biệt lớn'] += 1

        report = f"Điểm số tổng thể: {overall_score:.2f}/100\n\n"
        report += f"Số phần tử đã ghép cặp: {len(matched)}\n"
        report += f"Số phần tử bị thiếu trong thực tế (Missing): {len(missing)}\n"
        report += f"Số phần tử thừa trong thực tế (Extra): {len(extra)}\n\n"
        report += "Phân loại sự khác biệt (trong các cặp đã ghép):\n"
        for diff_type, count in diff_counts.items():
            report += f"  - {diff_type}: {count}\n"
        return report

    def visualize_difference_stats(self, analysis_result):
        """Hiển thị thống kê sự khác biệt."""
        # --- Code giữ nguyên, thêm .get() ---
        matched = analysis_result.get('matched_elements', [])
        missing = analysis_result.get('missing_elements', [])
        extra = analysis_result.get('extra_elements', [])

        diff_counts = {
            'Giống hệt': 0,
            'Gần giống': 0,
            'Khác biệt nhẹ': 0,
            'Khác biệt lớn': 0
        }
        for diff in matched:
            similarity = diff.get('similarity', 0.0)
            diff_type = diff.get('type')
            if diff_type is None:
                if similarity > 0.9: diff_type = "Giống hệt"
                elif similarity > 0.7: diff_type = "Gần giống"
                elif similarity > 0.5: diff_type = "Khác biệt nhẹ"
                else: diff_type = "Khác biệt lớn"

            if diff_type in diff_counts:
                 diff_counts[diff_type] += 1
            else:
                 diff_counts['Khác biệt lớn'] += 1

        labels_bar = list(diff_counts.keys())
        values_bar = list(diff_counts.values())

        plt.figure(figsize=(12, 5)) # Điều chỉnh figsize

        # Biểu đồ cột
        plt.subplot(1, 2, 1)
        if any(values_bar): # Chỉ vẽ nếu có dữ liệu
            bars = plt.bar(labels_bar, values_bar, color=['green', 'lightgreen', 'orange', 'red'])
            plt.title('Phân loại khác biệt (cặp ghép)')
            plt.ylabel('Số lượng')
            plt.xticks(rotation=45, ha='right')
            for bar in bars: # Thêm giá trị trên cột
                height = bar.get_height()
                if height > 0:
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        else:
            plt.text(0.5, 0.5, "Không có cặp ghép", ha='center', va='center')
            plt.title('Phân loại khác biệt (cặp ghép)')
            plt.xticks([])
            plt.yticks([])


        # Biểu đồ tròn
        plt.subplot(1, 2, 2)
        total_elements_in_scope = len(matched) + len(missing) + len(extra)
        if total_elements_in_scope > 0:
            labels_pie = ['Ghép cặp', 'Thiếu (Missing)', 'Thừa (Extra)']
            sizes_pie = [len(matched), len(missing), len(extra)]
            colors_pie = ['#66b3ff', '#ff9999', '#99ff99'] # Blue, Red, Greenish

            # Lọc ra các phần có kích thước > 0 để tránh lỗi wedge
            valid_indices = [i for i, size in enumerate(sizes_pie) if size > 0]
            labels_pie_valid = [labels_pie[i] for i in valid_indices]
            sizes_pie_valid = [sizes_pie[i] for i in valid_indices]
            colors_pie_valid = [colors_pie[i] for i in valid_indices]

            if sizes_pie_valid: # Nếu có dữ liệu hợp lệ để vẽ
                 plt.pie(sizes_pie_valid, labels=labels_pie_valid, colors=colors_pie_valid,
                       autopct=lambda p: '{:.1f}% ({:.0f})'.format(p, p * total_elements_in_scope / 100), # Hiển thị cả % và số lượng
                       startangle=90, wedgeprops={'edgecolor': 'white'})
                 plt.title('Phân bố phần tử UI')
                 plt.axis('equal')
            else: # Trường hợp tất cả bằng 0 (ít xảy ra)
                 plt.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center')
                 plt.title('Phân bố phần tử UI')
                 plt.axis('off')
        else:
             plt.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center')
             plt.title('Phân bố phần tử UI')
             plt.axis('off')

        plt.tight_layout(pad=2.0) # Tăng khoảng cách giữa các subplot
        plt.show()

    def display_results(self, img1, img2, elements1, elements2, matches, analysis_result, ssim_score, diff_heatmap, match_threshold=0.55):
        """
        Hiển thị tất cả kết quả với visualize tổng hợp.

        Args:
            match_threshold (float): Ngưỡng similarity tối thiểu để coi là khớp,
                                     truyền vào hàm draw_all_elements.
        """

        print("\n--- BẮT ĐẦU HIỂN THỊ KẾT QUẢ ---")

        #2. Phân đoạn UI
        print("Đang tạo ảnh kết quả phân đoạn...")  # Thêm log
        try:  # Thêm try-except
            segmented1 = self.draw_bounding_boxes(img1, elements1, show_indices=True)
            segmented2 = self.draw_bounding_boxes(img2, elements2, show_indices=True)
            plt.figure(figsize=(16, 10))  # Có thể dùng figsize khác
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(segmented1, cv2.COLOR_BGR2RGB))
            plt.title(f'Phân đoạn UI Design: {len(elements1)} thành phần')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(segmented2, cv2.COLOR_BGR2RGB))
            plt.title(f'Phân đoạn UI Thực Tế: {len(elements2)} thành phần')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Lỗi hiển thị ảnh phân đoạn: {e}")

        # 3. Hiển thị kết quả tổng hợp (khớp và không khớp)
        print("Đang tạo ảnh tổng hợp (Matched/Missing/Extra)...")
        try:
            vis1_all, vis2_all = self.draw_all_elements(img1, img2, elements1, elements2, matches, analysis_result, match_threshold)

            plt.figure(figsize=(16, 10)) # Tăng kích thước
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(vis1_all, cv2.COLOR_BGR2RGB))
            plt.title('UI Design (Green >0.9, Yellow Match, Red Missing)')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(vis2_all, cv2.COLOR_BGR2RGB))
            plt.title('UI Thực Tế (Green >0.9, Yellow Match, Red Extra)')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Lỗi hiển thị ảnh tổng hợp: {e}")


        # 4. Hiển thị SSIM heatmap (Giữ nguyên)
        print("Đang hiển thị SSIM heatmap...")
        try:
            plt.figure(figsize=(10, 6)) # Giảm kích thước heatmap
            plt.imshow(diff_heatmap, cmap='viridis') # diff_heatmap đã là ảnh màu BGR từ analyzer
            # Nếu diff_heatmap là map giá trị (0-1), dùng: plt.imshow(diff_heatmap, cmap='viridis')
            plt.title(f'Heatmap khác biệt cấu trúc (SSIM: {ssim_score:.3f})')
            plt.colorbar()
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Lỗi hiển thị SSIM heatmap: {e}")


        # 5. Hiển thị thống kê (Giữ nguyên)
        print("Đang hiển thị thống kê...")
        try:
            self.visualize_difference_stats(analysis_result)
        except Exception as e:
            print(f"Lỗi hiển thị thống kê: {e}")


        # 6. Tính điểm tổng thể (Giữ nguyên, cần tính lại các thành phần)
        print("Đang tính điểm tổng thể...")
        try:
            match_ratio = 0
            max_elements = max(len(elements1), len(elements2))
            if max_elements > 0:
                 match_ratio = len(matches) / max_elements

            # Khởi tạo lại analyzer để tính layout/color (hoặc đảm bảo nó stateless)
            temp_analyzer = DifferenceAnalyzer() # Giả sử stateless
            layout_consistency, _ = temp_analyzer.analyze_layout_consistency(matches)
            avg_color_diff, _ = temp_analyzer.analyze_color_differences(img1, img2)

            overall_score = temp_analyzer.compute_overall_difference_score(
                ssim_score, avg_color_diff, layout_consistency, match_ratio)
        except Exception as e:
            print(f"Lỗi tính điểm tổng thể: {e}")
            overall_score = -1 # Giá trị mặc định nếu lỗi


        # 7. Hiển thị báo cáo tổng hợp (Giữ nguyên)
        try:
            report = self.create_difference_report(analysis_result, overall_score)
            print("\n=== BÁO CÁO SO SÁNH UI ===")
            print(report)
        except Exception as e:
            print(f"Lỗi tạo báo cáo: {e}")

        print("--- KẾT THÚC HIỂN THỊ KẾT QUẢ ---")
