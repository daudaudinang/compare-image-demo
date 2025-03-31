# ui_comparator/visualization2.py

"""Module hiển thị kết quả trực quan."""

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import base64
import os
import json

from ui_comparator.analysis import DifferenceAnalyzer

matplotlib.use('TkAgg')

# --- Định nghĩa màu sắc (BGR) ---
# Giữ màu Cyan để kiểm tra từ lần trước, hoặc có thể đổi lại Vàng nếu muốn
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (255, 255, 0) # Giữ Cyan hoặc đổi lại (0, 255, 255) nếu muốn
COLOR_RED = (0, 0, 255)
# --------------------------------

class ResultVisualizer:
    """Hiển thị kết quả trực quan và tạo báo cáo HTML."""

    def __init__(self):
        pass

    # --- Các hàm vẽ đồ thị (draw_bounding_boxes, draw_all_elements, visualize_difference_stats) giữ nguyên ---
    def draw_bounding_boxes(self, img, elements, color=(0, 255, 0), show_indices=True):
        """Vẽ bounding box cho các thành phần UI."""
        result = img.copy()
        for i, element in enumerate(elements):
            try:
                bbox = element.get('bbox')
                if not isinstance(bbox, (tuple, list)) or len(bbox) != 4: continue
                x, y, w, h = map(int, map(float, bbox))
                if w > 0 and h > 0:
                    cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
                    if show_indices:
                        element_id = element.get('id', f'Idx-{i}')
                        cv2.putText(result, str(element_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            except (ValueError, TypeError, KeyError) as e:
                 print(f"Warning: Lỗi khi vẽ bbox cho element {element.get('id', i)}: {e}, bbox: {element.get('bbox')}")
        return result

    def draw_all_elements(self, img1, img2, elements1, elements2, matches, analysis_result, match_threshold=0.55):
        """Vẽ tất cả các phần tử lên ảnh gốc với màu sắc tương ứng."""
        vis1 = img1.copy(); vis2 = img2.copy()
        element1_map = {elem.get('id', f'D{i}'): elem for i, elem in enumerate(elements1)}
        element2_map = {elem.get('id', f'R{i}'): elem for i, elem in enumerate(elements2)}
        matched_ids1 = set(); matched_ids2 = set()

        for i, match in enumerate(matches):
            element1 = match.get('element1'); element2 = match.get('element2')
            similarity = match.get('similarity', 0.0)
            if element1 is None or element2 is None: continue
            color = COLOR_GREEN if similarity > 0.9 else COLOR_YELLOW
            id1 = element1.get('id', '?'); id2 = element2.get('id', '?')
            matched_ids1.add(id1); matched_ids2.add(id2)
            try: # Vẽ cho element 1
                bbox1 = element1.get('bbox')
                if bbox1:
                    x1, y1, w1, h1 = map(int, map(float, bbox1))
                    if w1 > 0 and h1 > 0:
                        cv2.rectangle(vis1, (x1, y1), (x1 + w1, y1 + h1), color, 2)
                        cv2.putText(vis1, f"{id1}({similarity:.2f})", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            except Exception as e: print(f"Lỗi vẽ match element 1 (ID {id1}): {e}")
            try: # Vẽ cho element 2
                bbox2 = element2.get('bbox')
                if bbox2:
                    x2, y2, w2, h2 = map(int, map(float, bbox2))
                    if w2 > 0 and h2 > 0:
                        cv2.rectangle(vis2, (x2, y2), (x2 + w2, y2 + h2), color, 2)
                        cv2.putText(vis2, f"{id2}({similarity:.2f})", (x2, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            except Exception as e: print(f"Lỗi vẽ match element 2 (ID {id2}): {e}")

        missing_elements = analysis_result.get('missing_elements', [])
        for element in missing_elements: # Vẽ missing
             try:
                 bbox = element.get('bbox'); id_missing = element.get('id', '?')
                 if bbox:
                     x, y, w, h = map(int, map(float, bbox))
                     if w > 0 and h > 0:
                         cv2.rectangle(vis1, (x, y), (x + w, y + h), COLOR_RED, 2)
                         cv2.putText(vis1, f"{id_missing} Missing", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_RED, 1)
             except Exception as e: print(f"Lỗi vẽ missing element (ID {id_missing}): {e}")
        extra_elements = analysis_result.get('extra_elements', [])
        for element in extra_elements: # Vẽ extra
             try:
                 bbox = element.get('bbox'); id_extra = element.get('id', '?')
                 if bbox:
                     x, y, w, h = map(int, map(float, bbox))
                     if w > 0 and h > 0:
                         cv2.rectangle(vis2, (x, y), (x + w, y + h), COLOR_RED, 2)
                         cv2.putText(vis2, f"{id_extra} Extra", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_RED, 1)
             except Exception as e: print(f"Lỗi vẽ extra element (ID {id_extra}): {e}")
        return vis1, vis2

    def visualize_difference_stats(self, analysis_result):
        """Hiển thị thống kê sự khác biệt (đồ thị)."""
        # --- Code giữ nguyên ---
        matched = analysis_result.get('matched_elements', [])
        missing = analysis_result.get('missing_elements', [])
        extra = analysis_result.get('extra_elements', [])
        diff_counts = {'Giống hệt': 0,'Gần giống': 0,'Khác biệt nhẹ': 0,'Khác biệt lớn': 0}
        for diff in matched:
            similarity = diff.get('similarity', 0.0)
            diff_type = diff.get('type')
            if diff_type is None:
                if similarity > 0.9: diff_type = "Giống hệt"
                elif similarity > 0.7: diff_type = "Gần giống"
                elif similarity > 0.5: diff_type = "Khác biệt nhẹ"
                else: diff_type = "Khác biệt lớn"
            if diff_type in diff_counts: diff_counts[diff_type] += 1
            else: diff_counts['Khác biệt lớn'] += 1
        labels_bar = list(diff_counts.keys()); values_bar = list(diff_counts.values())
        plt.figure("Thống kê khác biệt", figsize=(12, 5))
        plt.subplot(1, 2, 1) # Biểu đồ cột
        if any(values_bar):
            bars = plt.bar(labels_bar, values_bar, color=['green', 'lightgreen', 'orange', 'red'])
            plt.title('Phân loại khác biệt (cặp ghép)'); plt.ylabel('Số lượng'); plt.xticks(rotation=45, ha='right')
            for bar in bars:
                height = bar.get_height()
                if height > 0: plt.text(bar.get_x() + bar.get_width()/2., height,f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        else: plt.text(0.5, 0.5, "Không có cặp ghép", ha='center', va='center'); plt.title('Phân loại khác biệt (cặp ghép)'); plt.xticks([]); plt.yticks([])
        plt.subplot(1, 2, 2) # Biểu đồ tròn
        total_elements_in_scope = len(matched) + len(missing) + len(extra)
        if total_elements_in_scope > 0:
            labels_pie = ['Ghép cặp', 'Thiếu (Missing)', 'Thừa (Extra)']; sizes_pie = [len(matched), len(missing), len(extra)]
            colors_pie = ['#66b3ff', '#ff9999', '#99ff99']
            valid_indices = [i for i, size in enumerate(sizes_pie) if size > 0]
            labels_pie_valid = [labels_pie[i] for i in valid_indices]; sizes_pie_valid = [sizes_pie[i] for i in valid_indices]; colors_pie_valid = [colors_pie[i] for i in valid_indices]
            if sizes_pie_valid:
                 plt.pie(sizes_pie_valid, labels=labels_pie_valid, colors=colors_pie_valid, autopct=lambda p: '{:.1f}% ({:.0f})'.format(p, p * total_elements_in_scope / 100), startangle=90, wedgeprops={'edgecolor': 'white'})
                 plt.title('Phân bố phần tử UI'); plt.axis('equal')
            else: plt.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center'); plt.title('Phân bố phần tử UI'); plt.axis('off')
        else: plt.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center'); plt.title('Phân bố phần tử UI'); plt.axis('off')
        plt.tight_layout(pad=2.0); plt.show(block=False)

    # --- Hàm tạo báo cáo tổng hợp (in console) ---
    def create_difference_report(self, analysis_result, overall_score):
        """Tạo báo cáo về sự khác biệt (in ra console)."""
        # --- Code giữ nguyên ---
        matched = analysis_result.get('matched_elements', [])
        missing = analysis_result.get('missing_elements', [])
        extra = analysis_result.get('extra_elements', [])
        diff_counts = {'Giống hệt': 0,'Gần giống': 0,'Khác biệt nhẹ': 0,'Khác biệt lớn': 0}
        for diff in matched:
            similarity = diff.get('similarity', 0.0)
            diff_type = diff.get('type')
            if diff_type is None:
                if similarity > 0.9: diff_type = "Giống hệt"
                elif similarity > 0.7: diff_type = "Gần giống"
                elif similarity > 0.5: diff_type = "Khác biệt nhẹ"
                else: diff_type = "Khác biệt lớn"
            if diff_type in diff_counts: diff_counts[diff_type] += 1
            else: diff_counts['Khác biệt lớn'] += 1
        report = f"Điểm số tổng thể: {overall_score:.2f}/100\n\n"
        report += f"Số phần tử đã ghép cặp: {len(matched)}\n"
        report += f"Số phần tử bị thiếu trong thực tế (Missing): {len(missing)}\n"
        report += f"Số phần tử thừa trong thực tế (Extra): {len(extra)}\n\n"
        report += "Phân loại sự khác biệt (trong các cặp đã ghép):\n"
        for diff_type, count in diff_counts.items(): report += f"  - {diff_type}: {count}\n"
        return report

    # <<< HÀM TẠO BÁO CÁO HTML (SỬA LỖI MÀU) >>>
    def generate_html_report(self, vis_img1, vis_img2, analysis_result, ssim_score, overall_score, output_dir="report_output", filename="ui_comparison_report.html"):
        """Tạo file báo cáo HTML chứa kết quả so sánh và phân tích VLM (hỗ trợ JSON)."""
        vlm_analysis_list = analysis_result.get('vlm_details', [])
        matched_elements = analysis_result.get('matched_elements', [])
        missing_elements = analysis_result.get('missing_elements', [])
        extra_elements = analysis_result.get('extra_elements', [])

        print(f"\nĐang tạo báo cáo HTML: {filename}...")
        # print(f"DEBUG: Số lượng VLM details nhận được: {len(vlm_analysis_list)}") # Giữ lại debug nếu cần

        # --- Hàm helper để encode ảnh sang base64 (ĐÃ SỬA) ---
        def encode_image_to_base64_uri(img_np):
            if img_np is None or img_np.size == 0: return ""
            try:
                # <<< BỎ QUA CHUYỂN ĐỔI BGR->RGB Ở ĐÂY >>>
                # Giả định rằng img_np (là vis_img1/vis_img2) đã ở định dạng phù hợp
                # mà Matplotlib hiển thị đúng (thường là RGB).
                # Nếu Matplotlib hiển thị đúng, thì ảnh này có thể đã là RGB.
                # img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Bỏ dòng này

                # Encode trực tiếp sang PNG rồi base64
                # Truyền img_np (có thể là RGB) vào imencode
                retval, buffer = cv2.imencode('.png', img_np)
                # <<< KẾT THÚC THAY ĐỔI >>>

                if not retval: return ""
                base64_image = base64.b64encode(buffer).decode('utf-8')
                return f"data:image/png;base64,{base64_image}"
            except Exception as e:
                print(f"   Lỗi encode ảnh sang base64: {e}")
                return ""
        # --- Kết thúc hàm helper ---

        img1_base64 = encode_image_to_base64_uri(vis_img1)
        img2_base64 = encode_image_to_base64_uri(vis_img2)

        # --- Xây dựng nội dung HTML ---
        # (Phần CSS và cấu trúc HTML giữ nguyên như trước)
        html_content = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo cáo so sánh UI</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; line-height: 1.5; }}
        h1, h2 {{ color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px;}}
        .container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
        .image-container {{ text-align: center; flex: 1; min-width: 45%; }}
        .image-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; table-layout: fixed; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; word-wrap: break-word; }}
        th {{ background-color: #f2f2f2; }}
        td.vlm-analysis pre {{ background-color: #f8f8f8; border: 1px solid #eee; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-break: break-all; font-size: 0.9em; margin: 0; }}
        td.vlm-analysis .error {{ color: red; font-style: italic; }}
        .summary p {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>Báo cáo so sánh UI</h1>
    <div class="summary">
        <h2>Tóm tắt kết quả</h2>
        <p><strong>Điểm tương đồng cấu trúc (SSIM):</strong> {ssim_score:.3f}</p>
        <p><strong>Điểm khác biệt tổng thể (0-100, càng cao càng giống):</strong> {overall_score:.2f}</p>
        <p><strong>Số cặp ghép được:</strong> {len(matched_elements)}</p>
        <p><strong>Số phần tử thiếu (Missing):</strong> {len(missing_elements)}</p>
        <p><strong>Số phần tử thừa (Extra):</strong> {len(extra_elements)}</p>
    </div>
    <h2>So sánh trực quan</h2>
    <div class="container">
        <div class="image-container">
            <h3>UI Design (Kết quả khớp)</h3>
            <img src="{img1_base64}" alt="UI Design Visualization">
        </div>
        <div class="image-container">
            <h3>UI Thực Tế (Kết quả khớp)</h3>
            <img src="{img2_base64}" alt="UI Real Visualization">
        </div>
    </div>
    <h2>Phân tích chi tiết từ VLM (Gemini)</h2>
    """

        if vlm_analysis_list:
            html_content += """
    <table>
        <thead>
            <tr>
                <th style="width:15%;">Cặp ID</th>
                <th style="width:15%;">Similarity (%)</th>
                <th style="width:70%;">Phân tích VLM (JSON hoặc Lỗi)</th>
            </tr>
        </thead>
        <tbody>
    """
            # <<< DEBUG PRINT VẪN GIỮ LẠI NẾU CẦN >>>
            for idx, analysis_item in enumerate(vlm_analysis_list):
                # print(f"DEBUG: Processing VLM item {idx}: {analysis_item}")
                id1 = analysis_item.get('element1_id', '?')
                id2 = analysis_item.get('element2_id', '?')
                pair_id = f"{id1} vs {id2}"
                similarity = analysis_item.get('similarity', 0.0)
                similarity_str = f"{similarity*100:.1f}%"
                vlm_result = analysis_item.get('vlm_analysis', 'N/A')
                # print(f"DEBUG: VLM result for item {idx}: Type={type(vlm_result)}, Value='{str(vlm_result)[:100]}...'")

                # Kiểm tra và định dạng kết quả VLM
                analysis_content_html = ""
                if isinstance(vlm_result, dict):
                    try:
                        json_str_formatted = json.dumps(vlm_result, indent=2, ensure_ascii=False)
                        analysis_content_html = f"<pre>{json_str_formatted.replace('<', '&lt;').replace('>', '&gt;')}</pre>"
                    except Exception as format_err:
                        analysis_content_html = f"<span class='error'>Lỗi định dạng JSON: {format_err}</span><br><pre>{str(vlm_result).replace('<', '&lt;').replace('>', '&gt;')}</pre>"
                elif isinstance(vlm_result, str):
                    if vlm_result.startswith("Lỗi") or vlm_result == "Client chưa được cấu hình.":
                        analysis_content_html = f"<span class='error'>{vlm_result.replace('<', '&lt;').replace('>', '&gt;')}</span>"
                    elif vlm_result == 'N/A':
                         analysis_content_html = "N/A"
                    else:
                         analysis_content_html = f"<pre>{vlm_result.replace('<', '&lt;').replace('>', '&gt;')}</pre>"
                else:
                    analysis_content_html = str(vlm_result).replace('<', '&lt;').replace('>', '&gt;')

                html_content += f"""
            <tr>
                <td>{pair_id}</td>
                <td>{similarity_str}</td>
                <td class="vlm-analysis">{analysis_content_html}</td>
            </tr>
        """
            html_content += """
        </tbody>
    </table>
    """
        else:
            html_content += "<p>(Không có phân tích chi tiết từ VLM để hiển thị)</p>"

        html_content += """
</body>
</html>
        """

        # --- Lưu file HTML ---
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"Đã lưu báo cáo HTML thành công vào: {os.path.abspath(output_path)}")
            return os.path.abspath(output_path)
        except Exception as e:
            print(f"Lỗi khi lưu báo cáo HTML: {e}")
            return None
    # <<< KẾT THÚC HÀM >>>


    def display_results(self, img1, img2, elements1, elements2, matches, analysis_result, ssim_score, diff_heatmap, match_threshold=0.55):
        """
        Hiển thị các đồ thị Matplotlib và tạo báo cáo HTML.
        """
        print("\n--- BẮT ĐẦU HIỂN THỊ KẾT QUẢ ---")

        # 1. Ảnh gốc và phân đoạn (Matplotlib)
        print("Đang tạo ảnh gốc và phân đoạn...")
        try:
            segmented1 = self.draw_bounding_boxes(img1, elements1, show_indices=True)
            segmented2 = self.draw_bounding_boxes(img2, elements2, show_indices=True)
            plt.figure("Ảnh gốc và Phân đoạn", figsize=(16, 10))
            plt.subplot(2, 2, 1); plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)); plt.title('UI Design (Gốc)'); plt.axis('off')
            plt.subplot(2, 2, 2); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)); plt.title('UI Thực Tế (Gốc)'); plt.axis('off')
            plt.subplot(2, 2, 3); plt.imshow(cv2.cvtColor(segmented1, cv2.COLOR_BGR2RGB)); plt.title(f'Phân đoạn UI Design: {len(elements1)}'); plt.axis('off')
            plt.subplot(2, 2, 4); plt.imshow(cv2.cvtColor(segmented2, cv2.COLOR_BGR2RGB)); plt.title(f'Phân đoạn UI Thực Tế: {len(elements2)}'); plt.axis('off')
            plt.tight_layout(); plt.show(block=False)
        except Exception as e: print(f"Lỗi hiển thị ảnh gốc/phân đoạn: {e}")

        # 2. Kết quả tổng hợp (khớp và không khớp) (Matplotlib)
        print("Đang tạo ảnh tổng hợp (Matched/Missing/Extra)...")
        vis1_all, vis2_all = None, None
        try:
            vis1_all, vis2_all = self.draw_all_elements(img1, img2, elements1, elements2, matches, analysis_result, match_threshold)
            plt.figure("Kết quả khớp tổng hợp", figsize=(16, 8))
            plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(vis1_all, cv2.COLOR_BGR2RGB)); plt.title('UI Design (Kết quả khớp)'); plt.axis('off')
            plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(vis2_all, cv2.COLOR_BGR2RGB)); plt.title('UI Thực Tế (Kết quả khớp)'); plt.axis('off')
            plt.tight_layout(); plt.show(block=False)
        except Exception as e: print(f"Lỗi hiển thị ảnh tổng hợp: {e}")

        # 3. SSIM heatmap (Matplotlib)
        print("Đang hiển thị SSIM heatmap...")
        try:
            plt.figure("Heatmap SSIM", figsize=(8, 6))
            if diff_heatmap is not None:
                plt.imshow(cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB))
            else:
                plt.text(0.5, 0.5, "Không có heatmap", ha='center', va='center')
            plt.title(f'Heatmap khác biệt cấu trúc (SSIM: {ssim_score:.3f})'); plt.axis('off')
            plt.show(block=False)
        except Exception as e: print(f"Lỗi hiển thị SSIM heatmap: {e}")

        # 4. Thống kê (Matplotlib)
        print("Đang hiển thị thống kê...")
        try:
            self.visualize_difference_stats(analysis_result)
        except Exception as e: print(f"Lỗi hiển thị thống kê: {e}")

        # 5. Tính điểm tổng thể
        print("Đang tính điểm tổng thể...")
        overall_score = -1
        try:
            match_ratio = 0; max_elements = max(len(elements1), len(elements2))
            if max_elements > 0: match_ratio = len(matches) / max_elements
            temp_analyzer = DifferenceAnalyzer()
            layout_consistency, _ = temp_analyzer.analyze_layout_consistency(matches)
            avg_color_diff, _ = temp_analyzer.analyze_color_differences(img1, img2)
            overall_score = temp_analyzer.compute_overall_difference_score(ssim_score, avg_color_diff, layout_consistency, match_ratio)
        except Exception as e: print(f"Lỗi tính điểm tổng thể: {e}")

        # 6. Hiển thị báo cáo tổng hợp (in ra console)
        try:
            report = self.create_difference_report(analysis_result, overall_score)
            print("\n=== BÁO CÁO SO SÁNH UI (Console) ===")
            print(report)
        except Exception as e: print(f"Lỗi tạo báo cáo tổng hợp console: {e}")

        # 7. Tạo báo cáo HTML (đã cập nhật)
        html_report_path = None
        try:
            if vis1_all is not None and vis2_all is not None:
                 html_report_path = self.generate_html_report(
                     vis1_all, vis2_all, analysis_result, ssim_score, overall_score
                 )
            else:
                 print("Cảnh báo: Không thể tạo báo cáo HTML do thiếu ảnh visualize tổng hợp.")
        except Exception as e:
            print(f"Lỗi khi tạo báo cáo HTML: {e}")

        print("\n--- KẾT THÚC HIỂN THỊ KẾT QUẢ ---")
        if html_report_path:
             print(f"*** Báo cáo chi tiết VLM đã được lưu vào file HTML: {html_report_path}")
             print("*** Hãy mở file này bằng trình duyệt web để xem.")
        print(">>> Đang hiển thị các cửa sổ đồ thị. Đóng các cửa sổ để kết thúc chương trình.")
        plt.show() # Giữ các cửa sổ Matplotlib mở

