"""Module phân đoạn các thành phần UI."""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import paddleocr
from paddleocr import PaddleOCR

class UISegmenter:
    """Phân đoạn các thành phần UI."""
    
    def __init__(self):
        # Khởi tạo PaddleOCR (chỉ cần một lần)
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False, use_dilation=True, det_db_score_mode='slow')
    
    def segment(self, img):
        """Thực hiện phân đoạn UI đa phương pháp."""
        all_elements = []
        
        # # 1. Phát hiện logo - No detect logo
        # logo_elements = self.detect_logos(img)
        # all_elements.extend(logo_elements)
        
        # # 2. Phát hiện button
        # button_elements = self.detect_buttons(img)
        # all_elements.extend(button_elements)
        
        # 3. Phát hiện text với PaddleOCR
        text_elements = self.detect_text(img)
        print(f"Phát hiện {len(text_elements)} text elements")
        all_elements.extend(text_elements)
        
        # # 4. Phát hiện các thành phần UI khác
        # other_elements = self.segment_by_color(img)
        # all_elements.extend(other_elements)
        
        # 5. Lọc và phân loại các phần tử
        classified_elements = self.classify_all_elements(all_elements)
        
        # 6. Lọc các phần tử chồng lấp và thừa
        filtered_elements = self.filter_overlapping_elements(classified_elements)
        
        # 7. Gộp các text liền kề
        merged_text_elements = self.merge_text_elements(filtered_elements)

        print('merged_text_elements', merged_text_elements)
        
        return merged_text_elements
    
    def preprocess_img_for_ocr(self, img):
        """Tiền xử lý ảnh trước khi OCR."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(blurred)
        thresholded = cv2.adaptiveThreshold(
            enhanced_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return thresholded

    def detect_text(self, img):
        """Phát hiện text với PaddleOCR (kết hợp tiền xử lý)."""
        preprocessed_img = self.preprocess_img_for_ocr(img)
        result = self.ocr.ocr(preprocessed_img, cls=True)
        
        text_elements = []
        if result is not None:
            for line in result:
                if line is not None:
                    for item in line:
                        box = item[0]
                        text = item[1][0]
                        confidence = item[1][1]
                        
                        x = min(int(box[0][0]), int(box[1][0]), int(box[2][0]), int(box[3][0]))
                        y = min(int(box[0][1]), int(box[1][1]), int(box[2][1]), int(box[3][1]))
                        max_x = max(int(box[0][0]), int(box[1][0]), int(box[2][0]), int(box[3][0]))
                        max_y = max(int(box[0][1]), int(box[1][1]), int(box[2][1]), int(box[3][1]))
                        w = max_x - x
                        h = max_y - y

                        if confidence > 0.5:
                            y_safe = max(0, y)
                            x_safe = max(0, x)
                            h_safe = min(h, img.shape[0] - y_safe)
                            w_safe = min(w, img.shape[1] - x_safe)
                            
                            if h_safe > 0 and w_safe > 0:  # Kiểm tra vùng cắt hợp lệ
                                text_elements.append({
                                    'bbox': (x, y, w, h),
                                    'type': 'text',
                                    'text': text,
                                    'confidence': confidence,
                                    'contour': None,
                                    'roi': img[y_safe:y_safe+h_safe, x_safe:x_safe+w_safe].copy()
                                })
        
        return text_elements
    
    def detect_buttons(self, img):
        """Phát hiện button với nhiều thuật toán khác nhau."""
        button_elements = []
        
        # 1. Phát hiện button dựa trên màu sắc đồng nhất và hình dạng chữ nhật
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Phát hiện các vùng có màu sắc đồng nhất (button thường có màu đồng nhất)
        for channel in range(3):  # H, S, V
            for threshold in range(30, 240, 30):  # Thử nhiều ngưỡng
                mask = cv2.inRange(hsv[:,:,channel], threshold-15, threshold+15)
                
                # Tìm contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 1000 or area > 0.3 * img.shape[0] * img.shape[1]:
                        continue
                    
                    # Kiểm tra hình dạng (button thường là hình chữ nhật)
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Tỷ lệ w/h của button thường từ 2:1 đến 6:1
                    if 1.5 <= w/h <= 6 and h >= 30:
                        # Kiểm tra độ chữ nhật
                        rect_area = w * h
                        if area / rect_area > 0.8:  # Diện tích chiếm > 80% hình chữ nhật
                            # Lấy mẫu màu từ phần giữa button
                            center_x, center_y = x + w//2, y + h//2
                            margin = min(w, h) // 4
                            roi = img[center_y-margin:center_y+margin, center_x-margin:center_x+margin]
                            
                            if roi.size > 0:
                                # Tính độ đồng nhất màu sắc
                                std_dev = np.std(roi.reshape(-1, 3), axis=0).mean()
                                if std_dev < 20:  # Màu đồng nhất
                                    button_elements.append({
                                        'bbox': (x, y, w, h),
                                        'contour': contour,
                                        'type': 'button',
                                        'roi': img[y:y+h, x:x+w].copy()
                                    })
        
        # 2. Phát hiện button dựa trên cạnh và hình dáng
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000 or area > 0.3 * img.shape[0] * img.shape[1]:
                continue
            
            # Xấp xỉ đa giác
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Button thường có 4-8 cạnh (hình chữ nhật có bo tròn)
            if 4 <= len(approx) <= 8:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Kiểm tra tỷ lệ w/h
                if 1.5 <= w/h <= 6 and h >= 30:
                    button_elements.append({
                        'bbox': (x, y, w, h),
                        'contour': contour,
                        'type': 'button',
                        'roi': img[y:y+h, x:x+w].copy()
                    })
        
        # Loại bỏ các button trùng lặp
        return self.remove_duplicates(button_elements)
    
    def detect_logos(self, img):
        """Phát hiện logo với thuật toán chuyên biệt."""
        logos = []
        
        # Logo thường có tỷ lệ gần vuông, màu sắc nổi bật
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Tìm contour với nhiều phương pháp ngưỡng khác nhau
        for method in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
            for threshold in range(50, 200, 30):
                _, binary = cv2.threshold(blurred, threshold, 255, method)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    # Logo thường có kích thước vừa phải, không quá lớn hoặc quá nhỏ
                    if 500 < area < 20000:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Logo thường có tỷ lệ gần vuông
                        aspect_ratio = w / h
                        if 0.8 <= aspect_ratio <= 1.2:
                            # Kiểm tra độ phức tạp (logo thường có nhiều chi tiết)
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            solidity = float(area) / hull_area if hull_area > 0 else 0
                            
                            # Logo thường có solidity trung bình
                            if 0.5 <= solidity <= 0.9:
                                logos.append({
                                    'bbox': (x, y, w, h),
                                    'contour': contour,
                                    'type': 'logo',
                                    'roi': img[y:y+h, x:x+w].copy()
                                })
        
        # Loại bỏ logo trùng lặp
        return self.remove_duplicates(logos)
    
    def remove_duplicates(self, elements, overlap_threshold=0.7):
        """Loại bỏ phần tử trùng lặp."""
        if not elements:
            return []
            
        # Sắp xếp theo diện tích (lớn đến nhỏ)
        elements_sorted = sorted(elements, key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)
        
        filtered = []
        for element_i in elements_sorted:
            x_i, y_i, w_i, h_i = element_i['bbox']
            area_i = w_i * h_i
            
            is_duplicate = False
            for element_j in filtered:
                x_j, y_j, w_j, h_j = element_j['bbox']
                
                # Tính diện tích giao nhau
                x_overlap = max(0, min(x_i + w_i, x_j + w_j) - max(x_i, x_j))
                y_overlap = max(0, min(y_i + h_i, y_j + h_j) - max(y_i, y_j))
                area_overlap = x_overlap * y_overlap
                
                # Nếu chồng lấp nhiều, coi là trùng lặp
                if area_overlap / min(area_i, w_j * h_j) > overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(element_i)
        
        return filtered

    
    def segment_by_color(self, img):
        """Phân đoạn dựa trên màu sắc với độ nhạy cao hơn."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)
        
        # Tăng số lượng cụm màu để phát hiện phần tử nhỏ
        n_clusters = 12  # Tăng từ 8 lên 12
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels)
        labels = kmeans.labels_.reshape(img.shape[:2])
        
        color_elements = []
        for i in range(n_clusters):
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask[labels == i] = 255
            
            # Làm mịn mask
            kernel = np.ones((3, 3), np.uint8)  # Giảm kích thước kernel để giữ phần tử nhỏ
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Tìm contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # Giảm ngưỡng diện tích để bắt phần tử nhỏ (10 pixel trở lên)
                if area > 10 and area < 0.5 * img.shape[0] * img.shape[1]:
                    x, y, w, h = cv2.boundingRect(contour)
                    color_elements.append({
                        'bbox': (x, y, w, h),
                        'contour': contour,
                        'type': 'color_segment',
                        'roi': img[y:y+h, x:x+w].copy()
                    })
        
        return color_elements
    
    def detect_edges(self, img):
        """Phát hiện các UI element dựa trên cạnh."""
        # Chuyển sang grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Áp dụng GaussianBlur để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Sử dụng Canny với ngưỡng thấp để bắt nhiều cạnh hơn
        edges = cv2.Canny(blurred, 30, 150)  # Giảm ngưỡng dưới
        
        # Tăng cường cạnh
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Tìm contour
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        edge_elements = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Điều chỉnh ngưỡng cho phần tử nhỏ
            if area > 10 and area < 0.5 * img.shape[0] * img.shape[1]:
                # Lọc contour có hình dạng quá phức tạp
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.1:  # Giữ lại các contour có hình dạng hợp lý
                        x, y, w, h = cv2.boundingRect(contour)
                        edge_elements.append({
                            'bbox': (x, y, w, h),
                            'contour': contour,
                            'type': 'edge',
                            'roi': img[y:y+h, x:x+w].copy()
                        })
        
        return edge_elements
    
    def detect_icons(self, img):
        """Phát hiện icon đặc biệt như logo Apple, wifi, tín hiệu."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Tìm biểu tượng dựa trên ngưỡng sáng
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        icon_elements = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 500:  # Icon thường có diện tích nhỏ
                x, y, w, h = cv2.boundingRect(contour)
                # Kiểm tra tỷ lệ w/h (icon thường có tỷ lệ gần 1)
                if 0.5 < w/h < 2:
                    icon_elements.append({
                        'bbox': (x, y, w, h),
                        'contour': contour,
                        'type': 'icon',
                        'roi': img[y:y+h, x:x+w].copy()
                    })
        
        # Phát hiện icon tối trên nền sáng
        _, binary_inv = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours_inv, _ = cv2.findContours(binary_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_inv:
            area = cv2.contourArea(contour)
            if 10 < area < 500:
                x, y, w, h = cv2.boundingRect(contour)
                if 0.5 < w/h < 2:
                    # Kiểm tra xem đã có phần tử tương tự chưa
                    is_duplicate = False
                    for element in icon_elements:
                        x2, y2, w2, h2 = element['bbox']
                        if abs(x-x2) < max(w, w2)/2 and abs(y-y2) < max(h, h2)/2:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        icon_elements.append({
                            'bbox': (x, y, w, h),
                            'contour': contour,
                            'type': 'icon',
                            'roi': img[y:y+h, x:x+w].copy()
                        })
        
        return icon_elements
    
    def classify_element_type(self, element):
        """Phân loại loại phần tử UI (text, icon, button, etc.)."""
        x, y, w, h = element['bbox']
        roi = element['roi']
        
        # Nếu đã được OCR phát hiện
        if 'text' in element:
            return 'text'
        
        # Nếu đã được phân loại là button
        if element.get('type') == 'button':
            return 'button'
        
        # Phân loại dựa trên tỷ lệ và kích thước
        aspect_ratio = w / h
        
        # Icon thường có tỷ lệ gần 1:1
        if 0.8 <= aspect_ratio <= 1.2 and max(w, h) < 50:
            return 'icon'
        
        # Button thường là hình chữ nhật nằm ngang
        if 1.5 <= aspect_ratio <= 6 and min(w, h) >= 20:
            # Kiểm tra màu đồng nhất
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_roi], [0], None, [36], [0, 180])
            hist_norm = hist / hist.sum()
            
            if np.max(hist_norm) > 0.3:
                return 'button'
        
        # Phân tích màu sắc và texture
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Đếm pixel trắng/đen
        white_ratio = np.sum(binary == 255) / (w * h)
        
        # Text thường có tỷ lệ trắng/đen cân bằng
        if 0.3 <= white_ratio <= 0.7:
            return 'text'
        
        # Loại bỏ whitespace
        if white_ratio > 0.95:
            return 'whitespace'
        
        return 'unknown'
    
    def classify_all_elements(self, elements):
        """Phân loại tất cả các phần tử."""
        classified = []
        
        for element in elements:
            if 'type' not in element or element['type'] in ['color_segment', 'edge']:
                element_type = self.classify_element_type(element)
                element['type'] = element_type
                
                # Loại bỏ whitespace
                if element_type != 'whitespace':
                    classified.append(element)
            else:
                classified.append(element)
        
        return classified
    
    def filter_overlapping_elements(self, elements, overlap_threshold=0.6):
        """Lọc các phần tử trùng lặp và chồng lấp."""
        if not elements:
            return []
        
        # Ưu tiên theo thứ tự: text > button > icon > other
        type_priority = {'text': 3, 'button': 2, 'icon': 1, 'unknown': 0}
        
        # Sắp xếp theo ưu tiên và diện tích
        elements_sorted = sorted(elements, 
                               key=lambda x: (type_priority.get(x.get('type', 'unknown'), 0), 
                                             x['bbox'][2] * x['bbox'][3]), 
                               reverse=True)
        
        filtered = []
        for element_i in elements_sorted:
            x_i, y_i, w_i, h_i = element_i['bbox']
            area_i = w_i * h_i
            
            # Loại bỏ phần tử quá nhỏ
            if area_i < 16:  # Tăng ngưỡng để loại bỏ noise
                continue
                
            # Kiểm tra chồng lấp
            is_valid = True
            for element_j in filtered:
                x_j, y_j, w_j, h_j = element_j['bbox']
                
                # Tính phần giao nhau
                intersect_x = max(0, min(x_i + w_i, x_j + w_j) - max(x_i, x_j))
                intersect_y = max(0, min(y_i + h_i, y_j + h_j) - max(y_i, y_j))
                intersect_area = intersect_x * intersect_y
                
                # Tính tỷ lệ chồng lấp
                min_area = min(area_i, w_j * h_j)
                if min_area > 0:
                    overlap_ratio = intersect_area / min_area
                    
                    # Nếu chồng lấp nhiều
                    if overlap_ratio > overlap_threshold:
                        # Xử lý đặc biệt: text và button có thể chồng nhau
                        if (element_i['type'] == 'text' and element_j['type'] == 'button') or \
                           (element_i['type'] == 'button' and element_j['type'] == 'text'):
                            continue
                        
                        is_valid = False
                        break
            
            if is_valid:
                filtered.append(element_i)
        
        return filtered
    
    def merge_text_elements(self, elements):
        """Gộp các phần tử text bằng cách sử dụng đồ thị liên kết."""
        # Lọc danh sách các phần tử text
        text_elements = [e for e in elements if e['type'] == 'text']
        other_elements = [e for e in elements if e['type'] != 'text']
        
        if len(text_elements) <= 1:
            return text_elements + other_elements
        
        # Khởi tạo đồ thị liên kết
        adjacency_list = {i: [] for i in range(len(text_elements))}
        
        # Xây dựng đồ thị liên kết giữa các text gần nhau
        for i in range(len(text_elements)):
            bbox1 = text_elements[i]['bbox']
            x1, y1, w1, h1 = bbox1
            center1 = (x1 + w1 / 2, y1 + h1 / 2)
            
            for j in range(len(text_elements)):
                if i == j:
                    continue
                
                bbox2 = text_elements[j]['bbox']
                x2, y2, w2, h2 = bbox2
                center2 = (x2 + w2 / 2, y2 + h2 / 2)
                
                # Khoảng cách ngang/dọc
                horizontal_distance = abs(center1[0] - center2[0])
                vertical_distance = abs(center1[1] - center2[1])
                
                # Điều kiện gộp: gần nhau về không gian
                if horizontal_distance < max(w1, w2) * 1.5 and vertical_distance < max(h1, h2) * 1.5:
                    adjacency_list[i].append(j)
        
        # Gộp các thành phần liên thông
        visited = set()
        merged = []
        
        def dfs(node, component):
            """Duyệt DFS để tìm thành phần liên thông."""
            stack = [node]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                for neighbor in adjacency_list[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        for i in range(len(text_elements)):
            if i not in visited:
                component = []
                dfs(i, component)
                
                # Gộp các text trong cùng thành phần liên thông
                merged_text = ' '.join(text_elements[idx]['text'] for idx in sorted(component))
                
                # Tính bounding box bao quanh tất cả
                min_x = min(text_elements[idx]['bbox'][0] for idx in component)
                min_y = min(text_elements[idx]['bbox'][1] for idx in component)
                max_x = max(text_elements[idx]['bbox'][0] + text_elements[idx]['bbox'][2] for idx in component)
                max_y = max(text_elements[idx]['bbox'][1] + text_elements[idx]['bbox'][3] for idx in component)
                
                merged_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
                merged_element = {
                    'bbox': merged_bbox,
                    'type': 'text',
                    'text': merged_text,
                    'contour': None,
                    'roi': None  # Có thể trích xuất ROI tùy ý
                }
                merged.append(merged_element)
        
        # Kết hợp text đã gộp và các phần tử khác
        return merged + other_elements









