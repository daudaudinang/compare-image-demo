"""Module hiển thị kết quả trực quan."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Sử dụng backend TkAgg thay vì Qt

class ResultVisualizer:
    """Hiển thị kết quả trực quan."""
    
    def __init__(self):
        pass
    
    def draw_bounding_boxes(self, img, elements, color=(0, 255, 0), show_indices=True):
        """Vẽ bounding box cho các thành phần UI."""
        result = img.copy()
        
        for i, element in enumerate(elements):
            x, y, w, h = element['bbox']
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            
            if show_indices:
                cv2.putText(result, f"UI-{i}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result
    
    def draw_matches(self, img1, img2, matches, show_similarity=True):
        """Vẽ các cặp ghép."""
        result1 = img1.copy()
        result2 = img2.copy()
        
        # Tạo bảng màu dựa trên độ tương đồng
        def get_color(similarity):
            # Màu chuyển từ đỏ (khác biệt) sang xanh lá (giống nhau)
            r = int(255 * (1 - similarity))
            g = int(255 * similarity)
            b = 0
            return (b, g, r)  # OpenCV sử dụng BGR
        
        for i, match in enumerate(matches):
            element1 = match['element1']
            element2 = match['element2']
            similarity = match['similarity']
            
            x1, y1, w1, h1 = element1['bbox']
            x2, y2, w2, h2 = element2['bbox']
            
            # Màu dựa trên độ tương đồng
            color = get_color(similarity)
            
            # Vẽ bounding box
            cv2.rectangle(result1, (x1, y1), (x1+w1, y1+h1), color, 2)
            cv2.rectangle(result2, (x2, y2), (x2+w2, y2+h2), color, 2)
            
            # Hiển thị số thứ tự
            cv2.putText(result1, f"{i}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(result2, f"{i}", (x2, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Hiển thị độ tương đồng
            if show_similarity:
                sim_text = f"{similarity:.2f}"
                cv2.putText(result1, sim_text, (x1, y1+h1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(result2, sim_text, (x2, y2+h2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result1, result2
    
    def mark_missing_elements(self, img, elements, color=(0, 0, 255)):
        """Đánh dấu các phần tử bị thiếu."""
        result = img.copy()
        
        for element in elements:
            x, y, w, h = element['bbox']
            
            # Vẽ các đường chéo
            cv2.line(result, (x, y), (x+w, y+h), color, 2)
            cv2.line(result, (x+w, y), (x, y+h), color, 2)
            
            # Vẽ bounding box
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        
        return result
    
    def create_difference_report(self, analysis_result, overall_score):
        """Tạo báo cáo về sự khác biệt."""
        matched = analysis_result['matched_elements']
        missing = analysis_result['missing_elements']
        extra = analysis_result['extra_elements']
        
        # Tính số lượng phần tử theo loại khác biệt
        diff_counts = {
            'Giống hệt': 0,
            'Gần giống': 0,
            'Khác biệt nhẹ': 0,
            'Khác biệt lớn': 0
        }
        
        for diff in matched:
            diff_type = diff['type']
            diff_counts[diff_type] += 1
        
        # Tạo báo cáo
        report = f"Điểm số tổng thể: {overall_score:.2f}/100\n\n"
        report += f"Số phần tử đã ghép cặp: {len(matched)}\n"
        report += f"Số phần tử bị thiếu trong thực tế: {len(missing)}\n"
        report += f"Số phần tử thừa trong thực tế: {len(extra)}\n\n"
        
        report += "Phân loại sự khác biệt:\n"
        for diff_type, count in diff_counts.items():
            report += f"  - {diff_type}: {count}\n"
        
        return report
    
    def visualize_difference_stats(self, analysis_result):
        """Hiển thị thống kê sự khác biệt."""
        matched = analysis_result['matched_elements']
        missing = analysis_result['missing_elements']
        extra = analysis_result['extra_elements']
        
        # Tính số lượng phần tử theo loại khác biệt
        diff_counts = {
            'Giống hệt': 0,
            'Gần giống': 0,
            'Khác biệt nhẹ': 0,
            'Khác biệt lớn': 0
        }
        
        for diff in matched:
            diff_type = diff['type']
            diff_counts[diff_type] += 1
        
        # Tạo biểu đồ
        labels = list(diff_counts.keys())
        values = list(diff_counts.values())
        
        plt.figure(figsize=(10, 6))
        
        # Biểu đồ cột
        plt.subplot(1, 2, 1)
        bars = plt.bar(labels, values, color=['green', 'lightgreen', 'orange', 'red'])
        plt.title('Phân loại sự khác biệt')
        plt.ylabel('Số lượng phần tử')
        plt.xticks(rotation=45, ha='right')
        
        # Thêm giá trị trên cột
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')
        
        # Biểu đồ tròn
        plt.subplot(1, 2, 2)
        total_elements = len(matched) + len(missing) + len(extra)
        
        if total_elements > 0:
            labels = ['Đã ghép cặp', 'Thiếu trong thực tế', 'Thừa trong thực tế']
            sizes = [len(matched), len(missing), len(extra)]
            colors = ['#66b3ff', '#ff9999', '#99ff99']
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                  startangle=90, wedgeprops={'edgecolor': 'white'})
            plt.title('Phân bố phần tử UI')
            plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    def display_results(self, img1, img2, elements1, elements2, matches, analysis_result, ssim_score, diff_heatmap):
        """Hiển thị tất cả kết quả."""
        # 1. Ảnh gốc
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title('UI Design')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.title('UI Thực Tế')
        plt.axis('off')
        
        # 2. Phân đoạn UI
        segmented1 = self.draw_bounding_boxes(img1, elements1)
        segmented2 = self.draw_bounding_boxes(img2, elements2)
        
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(segmented1, cv2.COLOR_BGR2RGB))
        plt.title(f'Phân đoạn UI Design: {len(elements1)} thành phần')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(segmented2, cv2.COLOR_BGR2RGB))
        plt.title(f'Phân đoạn UI Thực Tế: {len(elements2)} thành phần')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 3. Hiển thị các cặp ghép
        if matches:
            match_img1, match_img2 = self.draw_matches(img1, img2, matches)
            
            plt.figure(figsize=(15, 7))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(match_img1, cv2.COLOR_BGR2RGB))
            plt.title('Các phần tử được ghép cặp (Design)')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(match_img2, cv2.COLOR_BGR2RGB))
            plt.title('Các phần tử được ghép cặp (Thực tế)')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # 4. Hiển thị các phần tử thiếu/thừa
        missing_elements = analysis_result['missing_elements']
        extra_elements = analysis_result['extra_elements']
        
        if missing_elements or extra_elements:
            plt.figure(figsize=(15, 7))
            
            if missing_elements:
                missing_img = self.mark_missing_elements(img1, missing_elements)
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(missing_img, cv2.COLOR_BGR2RGB))
                plt.title(f'Phần tử thiếu trong UI thực tế: {len(missing_elements)}')
                plt.axis('off')
            
            if extra_elements:
                extra_img = self.mark_missing_elements(img2, extra_elements, color=(255, 0, 0))
                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(extra_img, cv2.COLOR_BGR2RGB))
                plt.title(f'Phần tử thừa trong UI thực tế: {len(extra_elements)}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # 5. Hiển thị SSIM heatmap
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB))
        plt.title(f'Heatmap sự khác biệt (SSIM: {ssim_score:.2f})')
        plt.colorbar()
        plt.axis('off')
        plt.show()
        
        # 6. Hiển thị thống kê
        self.visualize_difference_stats(analysis_result)
        
        # 7. Tính điểm tổng thể
        match_ratio = len(matches) / max(len(elements1), len(elements2)) if max(len(elements1), len(elements2)) > 0 else 0
        
        layout_analyzer = DifferenceAnalyzer()
        layout_consistency, _ = layout_analyzer.analyze_layout_consistency(matches)
        color_diff, _ = layout_analyzer.analyze_color_differences(img1, img2)
        
        overall_score = layout_analyzer.compute_overall_difference_score(
            ssim_score, color_diff, layout_consistency, match_ratio)
        
        # 8. Hiển thị báo cáo tổng hợp
        report = self.create_difference_report(analysis_result, overall_score)
        print("\n=== BÁO CÁO SO SÁNH UI ===")
        print(report)
