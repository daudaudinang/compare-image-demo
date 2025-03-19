"""Module phân tích sự khác biệt."""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class DifferenceAnalyzer:
    """Phân tích sự khác biệt giữa UI."""
    
    def __init__(self):
        pass
    
    def compare_structure(self, img1, img2):
        """So sánh cấu trúc tổng thể với SSIM."""
        # Đảm bảo hai ảnh có cùng kích thước
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if h1 != h2 or w1 != w2:
            img2 = cv2.resize(img2, (w1, h1))
        
        # Chuyển sang grayscale
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Tính SSIM
        score, diff_map = ssim(img1_gray, img2_gray, full=True)
        
        # Tạo heatmap
        diff_map_normalized = (diff_map * 255).astype(np.uint8)
        diff_heatmap = cv2.applyColorMap(255 - diff_map_normalized, cv2.COLORMAP_JET)
        
        return score, diff_map, diff_heatmap
    
    def analyze_element_differences(self, matches, elements1, elements2):
        """Phân tích sự khác biệt giữa các thành phần UI."""
        # Xác định các element đã được ghép
        matched_indices1 = set()
        matched_indices2 = set()
        
        for match in matches:
            element1 = match['element1']
            element2 = match['element2']
            
            idx1 = next((i for i, e in enumerate(elements1) if e == element1), None)
            idx2 = next((i for i, e in enumerate(elements2) if e == element2), None)
            
            if idx1 is not None:
                matched_indices1.add(idx1)
            if idx2 is not None:
                matched_indices2.add(idx2)
        
        # Xác định element chưa được ghép
        unmatched_elements1 = [elem for i, elem in enumerate(elements1) if i not in matched_indices1]
        unmatched_elements2 = [elem for i, elem in enumerate(elements2) if i not in matched_indices2]
        
        # Phân loại sự khác biệt
        difference_types = []
        for match in matches:
            similarity = match['similarity']
            
            if similarity > 0.9:
                difference_type = "Giống hệt"
            elif similarity > 0.7:
                difference_type = "Gần giống"
            elif similarity > 0.5:
                difference_type = "Khác biệt nhẹ"
            else:
                difference_type = "Khác biệt lớn"
            
            # Phân tích chi tiết về sự khác biệt
            element1 = match['element1']
            element2 = match['element2']
            
            # So sánh kích thước
            x1, y1, w1, h1 = element1['bbox']
            x2, y2, w2, h2 = element2['bbox']
            
            size_change = (w2*h2 - w1*h1) / (w1*h1) if w1*h1 > 0 else float('inf')
            
            # So sánh vị trí
            center1_x, center1_y = x1 + w1/2, y1 + h1/2
            center2_x, center2_y = x2 + w2/2, y2 + h2/2
            
            position_shift = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
            
            # Lưu chi tiết khác biệt
            difference_detail = {
                'type': difference_type,
                'similarity': similarity,
                'size_change': size_change,
                'position_shift': position_shift,
                'element1': element1,
                'element2': element2
            }
            
            difference_types.append(difference_detail)
        
        return {
            'matched_elements': difference_types,
            'missing_elements': unmatched_elements1,
            'extra_elements': unmatched_elements2
        }
    
    def analyze_color_differences(self, img1, img2):
        """Phân tích sự khác biệt về màu sắc."""
        # Đảm bảo hai ảnh có cùng kích thước
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if h1 != h2 or w1 != w2:
            img2 = cv2.resize(img2, (w1, h1))
        
        # Chuyển sang không gian màu LAB
        lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
        
        # Tính sự khác biệt cho từng kênh
        l_diff = np.abs(lab1[:,:,0].astype(float) - lab2[:,:,0].astype(float))
        a_diff = np.abs(lab1[:,:,1].astype(float) - lab2[:,:,1].astype(float))
        b_diff = np.abs(lab1[:,:,2].astype(float) - lab2[:,:,2].astype(float))
        
        # Tổng hợp sự khác biệt
        color_diff = (l_diff + a_diff + b_diff) / 3
        
        # Tính điểm trung bình
        avg_color_diff = np.mean(color_diff)
        max_color_diff = np.max(color_diff)
        
        # Tạo bản đồ nhiệt
        color_diff_normalized = (color_diff * 255 / max_color_diff).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(color_diff_normalized, cv2.COLORMAP_JET)
        
        return avg_color_diff, color_heatmap
    
    def analyze_layout_consistency(self, matches):
        """Phân tích độ nhất quán của layout."""
        if not matches:
            return 0.0, {}
        
        # Tính vector dịch chuyển cho từng cặp
        vectors = []
        for match in matches:
            element1 = match['element1']
            element2 = match['element2']
            
            x1, y1, w1, h1 = element1['bbox']
            x2, y2, w2, h2 = element2['bbox']
            
            # Tọa độ trung tâm
            center1_x, center1_y = x1 + w1/2, y1 + h1/2
            center2_x, center2_y = x2 + w2/2, y2 + h2/2
            
            # Vector dịch chuyển
            dx = center2_x - center1_x
            dy = center2_y - center1_y
            
            vectors.append((dx, dy))
        
        # Tính vector trung bình
        avg_dx = sum(v[0] for v in vectors) / len(vectors)
        avg_dy = sum(v[1] for v in vectors) / len(vectors)
        
        # Tính độ phân tán (sai số so với vector trung bình)
        deviations = [np.sqrt((dx - avg_dx)**2 + (dy - avg_dy)**2) for dx, dy in vectors]
        avg_deviation = sum(deviations) / len(deviations)
        
        # Độ nhất quán (càng gần 1 càng nhất quán)
        max_deviation = 100  # Giá trị tối đa cho phép
        consistency = max(0, 1 - avg_deviation / max_deviation)
        
        return consistency, {
            'average_shift': (avg_dx, avg_dy),
            'average_deviation': avg_deviation
        }
    
    def compute_overall_difference_score(self, ssim_score, color_diff, layout_consistency, match_ratio):
        """Tính điểm khác biệt tổng thể."""
        # Chuẩn hóa các thành phần
        ssim_score_norm = max(0, min(1, ssim_score))
        color_diff_norm = max(0, min(1, 1 - color_diff / 100))  # Chuẩn hóa về [0, 1]
        
        # Trọng số
        weights = {
            'ssim': 0.3,
            'color': 0.2,
            'layout': 0.2,
            'match_ratio': 0.3
        }
        
        # Tính điểm tổng hợp (càng cao càng giống nhau)
        similarity_score = (
            weights['ssim'] * ssim_score_norm +
            weights['color'] * color_diff_norm +
            weights['layout'] * layout_consistency +
            weights['match_ratio'] * match_ratio
        )
        
        return similarity_score * 100  # Đổi sang thang điểm 100
