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
        self.ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)

    def segment(self, img):
        """Thực hiện phân đoạn UI đa phương pháp."""
        # 1. Phát hiện text với PaddleOCR
        text_elements = self.detect_text(img)

        # 2. Phát hiện biên dựa trên Canny edge detection
        edge_elements = self.detect_edges(img)

        # 3. Gộp text
        merged_texts = self.merge_text_elements(text_elements)

        # 4. Kết hợp tất cả elements
        all_elements = merged_texts + edge_elements

        # 5. Lọc các elements chồng lấn
        merged_elements = self.merge_hierarchical_boxes(all_elements)

        filtered_elements = self.filter_overlapping_elements(merged_elements)

        return filtered_elements
    
    def detect_text(self, img):
        """Phát hiện text với PaddleOCR."""
        result = self.ocr.ocr(img, cls=True)
        
        text_elements = []
        if result is not None:
            for line in result:
                if line is not None:
                    for item in line:
                        box = item[0]  # Bounding box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text = item[1][0]  # Text content
                        confidence = item[1][1]  # Confidence score
                        
                        # Chuyển box thành x,y,w,h
                        x = min(int(box[0][0]), int(box[1][0]), int(box[2][0]), int(box[3][0]))
                        y = min(int(box[0][1]), int(box[1][1]), int(box[2][1]), int(box[3][1]))
                        max_x = max(int(box[0][0]), int(box[1][0]), int(box[2][0]), int(box[3][0]))
                        max_y = max(int(box[0][1]), int(box[1][1]), int(box[2][1]), int(box[3][1]))
                        w = max_x - x
                        h = max_y - y
                        
                        # Lưu kết quả với độ tin cậy cao
                        if confidence > 0.5:
                            # Tạo contour từ box
                            contour = np.array([
                                [int(box[0][0]), int(box[0][1])],
                                [int(box[1][0]), int(box[1][1])],
                                [int(box[2][0]), int(box[2][1])],
                                [int(box[3][0]), int(box[3][1])]
                            ], dtype=np.int32).reshape((-1, 1, 2))
                            
                            # Đảm bảo ROI không vượt quá kích thước ảnh
                            y_safe = max(0, y)
                            x_safe = max(0, x)
                            h_safe = min(h, img.shape[0]-y_safe)
                            w_safe = min(w, img.shape[1]-x_safe)
                            
                            # Kiểm tra xem ROI có hợp lệ không
                            if h_safe > 0 and w_safe > 0:
                                text_elements.append({
                                    'bbox': (x, y, w, h),
                                    'contour': contour,
                                    'text': text,
                                    'confidence': confidence,
                                    'type': 'text',
                                    'roi': img[y_safe:y_safe+h_safe, x_safe:x_safe+w_safe].copy()
                                })
        
        return text_elements

    def merge_hierarchical_boxes(self, boxes, containment_threshold=0.7, overlap_threshold=0.5):
        """
        Gộp các bounding box theo quan hệ chứa nhau hoặc chồng lấn.
        - Nếu box A chứa box B hoặc có phần giao lớn hơn ngưỡng overlap_threshold, thì A và B sẽ được gộp.
        - containment_threshold: ngưỡng xác định box A chứa box B.
        - overlap_threshold: ngưỡng xác định phần giao lớn giữa hai box.
        """
        if len(boxes) <= 1:
            return boxes

        # Sắp xếp boxes theo diện tích (lớn đến nhỏ)
        sorted_boxes = sorted(boxes, key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)

        # Danh sách các nhóm đã gộp
        merged_boxes = []

        # Kiểm tra quan hệ chứa nhau hoặc chồng lấn
        while len(sorted_boxes) > 0:
            # Lấy box đầu tiên ra khỏi danh sách
            current_box = sorted_boxes.pop(0)
            current_group = [current_box]

            i = 0
            while i < len(sorted_boxes):
                candidate_box = sorted_boxes[i]

                rect_a = current_box['bbox']
                rect_b = candidate_box['bbox']

                # Tính phần giao nhau
                x1_a, y1_a, w_a, h_a = rect_a
                x1_b, y1_b, w_b, h_b = rect_b

                intersect_x1 = max(x1_a, x1_b)
                intersect_y1 = max(y1_a, y1_b)
                intersect_x2 = min(x1_a + w_a, x1_b + w_b)
                intersect_y2 = min(y1_a + h_a, y1_b + h_b)

                if intersect_x2 <= intersect_x1 or intersect_y2 <= intersect_y1:
                    i += 1
                    continue

                # Tính diện tích phần giao
                intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)

                # Diện tích của candidate_box
                area_b = w_b * h_b

                # Kiểm tra quan hệ chứa nhau hoặc chồng lấn
                containment_ratio = intersect_area / area_b
                overlap_ratio = intersect_area / (w_a * h_a + area_b - intersect_area)

                if containment_ratio > containment_threshold or overlap_ratio > overlap_threshold:
                    current_group.append(candidate_box)
                    sorted_boxes.pop(i)
                else:
                    i += 1

            # Tạo bounding box mới bao quanh tất cả các box trong nhóm
            x_min = min(box['bbox'][0] for box in current_group)
            y_min = min(box['bbox'][1] for box in current_group)
            x_max = max(box['bbox'][0] + box['bbox'][2] for box in current_group)
            y_max = max(box['bbox'][1] + box['bbox'][3] for box in current_group)

            w_new = x_max - x_min
            h_new = y_max - y_min

            merged_box = {
                'bbox': (x_min, y_min, w_new, h_new),
                'contour': None,
                'area': w_new * h_new,
                'method': 'merged',
                'merged_count': len(current_group),
            }

            merged_boxes.append(merged_box)

        print(f"Gộp phân cấp và chồng lấn: từ {len(boxes)} xuống còn {len(merged_boxes)} UI elements")
        return merged_boxes

    def detect_edges(self, img):
        """Phát hiện các UI element dựa trên cạnh kết hợp đa phương pháp tăng cường độ tương phản."""
        import cv2
        import numpy as np

        # Định nghĩa hàm tính IoU
        def calculate_iou(bbox1, bbox2):
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2

            # Tính tọa độ của phần giao nhau
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)

            # Nếu không có phần giao nhau
            if x_right < x_left or y_bottom < y_top:
                return 0.0

            # Tính diện tích phần giao nhau
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # Tính diện tích của từng bbox
            bbox1_area = w1 * h1
            bbox2_area = w2 * h2

            # Tính IoU
            union_area = bbox1_area + bbox2_area - intersection_area
            iou = intersection_area / float(union_area)
            return iou

        # Hàm áp dụng gamma correction
        def apply_gamma(gray, gamma=1.0):
            # Normalization về khoảng [0,1]
            normalized = gray / 255.0

            # Áp dụng gamma correction
            corrected = np.power(normalized, gamma)

            # Chuyển về khoảng [0,255]
            output = np.uint8(corrected * 255)

            return output

        # Hàm phát hiện contours
        def detect_contours(img_gray, method_name, low_threshold=30, high_threshold=150):
            # Áp dụng GaussianBlur để giảm nhiễu
            kernel_size = (5, 5)
            blurred = cv2.GaussianBlur(img_gray, kernel_size, 0)

            # Sử dụng Canny để phát hiện cạnh
            edges = cv2.Canny(blurred, low_threshold, high_threshold)

            # Tăng cường cạnh với morphology
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)

            # Tìm contour
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Lọc contour
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)

                # Lọc theo diện tích
                if area > 100 and area < 0.3 * img.shape[0] * img.shape[1]:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Đối với Gamma 5.0, lọc bỏ contour lớn ở biên
                    border_valid = True
                    if method_name == "Gamma 5.0" and (
                            (x <= 5 and w > img.shape[1] * 0.7) or
                            (y <= 5 and h > img.shape[0] * 0.7)):
                        border_valid = False

                    if border_valid:
                        # Đảm bảo ROI không vượt quá kích thước ảnh
                        y_safe = max(0, y)
                        x_safe = max(0, x)
                        h_safe = min(h, img.shape[0] - y_safe)
                        w_safe = min(w, img.shape[1] - x_safe)

                        # Kiểm tra xem ROI có hợp lệ không
                        if h_safe > 0 and w_safe > 0:
                            # Tạo đối tượng element mà không lưu contour để tránh lỗi so sánh
                            # Lưu contour_id thay vì contour để có thể phân biệt
                            filtered_contours.append({
                                'bbox': (x, y, w, h),
                                'contour_id': id(contour),  # Dùng id để xác định contour
                                'type': 'edge',
                                'method': method_name,
                                'roi': img[y_safe:y_safe + h_safe, x_safe:x_safe + w_safe].copy()
                            })

            return filtered_contours

        # Chuyển sang grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Áp dụng các giá trị gamma
        gamma_07 = apply_gamma(gray, 0.7)  # Gamma thấp - tốt cho vùng tối
        gamma_25 = apply_gamma(gray, 2.5)  # Gamma cao - tốt cho vùng sáng có độ tương phản thấp
        gamma_50 = apply_gamma(gray, 4.0)  # Gamma rất cao - tốt cho một số UI elements khó phát hiện

        # Phát hiện contours với từng phương pháp và ngưỡng tương ứng
        original_contours = detect_contours(gray, "Original", 30, 150)
        gamma07_contours = detect_contours(gamma_07, "Gamma 0.7", 30, 150)
        gamma25_contours = detect_contours(gamma_25, "Gamma 2.5", 20, 100)
        gamma50_contours = detect_contours(gamma_50, "Gamma 5.0", 20, 100)

        # Kết hợp tất cả contours
        all_contours = []
        all_contours.extend(original_contours)
        all_contours.extend(gamma07_contours)
        all_contours.extend(gamma25_contours)
        all_contours.extend(gamma50_contours)

        # Lọc các contours trùng lặp dựa trên IoU
        # Không lưu contour trong các đối tượng element để tránh lỗi so sánh
        filtered_contours = []
        for contour in all_contours:
            # Kiểm tra xem contour này đã có trong kết quả chưa
            is_new = True
            current_bbox = contour['bbox']

            for existing in filtered_contours:
                if calculate_iou(current_bbox, existing['bbox']) > 0.5:
                    is_new = False
                    break

            if is_new:
                filtered_contours.append(contour)

        return filtered_contours

    def filter_overlapping_elements(self, elements):
        """
        Lọc các elements chồng lấn, ưu tiên giữ lại text elements.
        """
        if not elements:
            return []

        # Tách text elements và non-text elements
        text_elements = [e for e in elements if e.get('type') == 'text']
        non_text_elements = [e for e in elements if e.get('type') != 'text']

        # Sắp xếp text_elements theo diện tích (lớn đến nhỏ) - nếu cần
        text_elements.sort(key=lambda e: e['bbox'][2] * e['bbox'][3], reverse=True)

        # Gộp text và non-text, với text ở đầu
        sorted_elements = text_elements + non_text_elements

        # Lọc các elements nằm hoàn toàn trong elements khác
        result = []
        used = set()

        for i in range(len(sorted_elements)):
            if i in used:
                continue

            current = sorted_elements[i]
            result.append(current)
            used.add(i)

            # Kiểm tra các elements nhỏ hơn
            for j in range(i + 1, len(sorted_elements)):
                if j in used:
                    continue

                if self.is_contained(current['bbox'], sorted_elements[j]['bbox']):
                    used.add(j)

        return result

    def is_contained(self, bbox1, bbox2, threshold=0.9):
        """Kiểm tra bbox2 có nằm gần như hoàn toàn trong bbox1 không"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Tính toạ độ giao nhau
        intersect_x1 = max(x1, x2)
        intersect_y1 = max(y1, y2)
        intersect_x2 = min(x1 + w1, x2 + w2)
        intersect_y2 = min(y1 + h1, y2 + h2)

        # Nếu không có phần giao
        if intersect_x2 <= intersect_x1 or intersect_y2 <= intersect_y1:
            return False

        # Tính diện tích giao nhau
        intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)

        # Diện tích của bbox2
        area2 = w2 * h2

        # Nếu phần giao chiếm hơn threshold% diện tích của bbox2
        return intersect_area >= threshold * area2

    def merge_adjacent_texts(self, elements):
        """Gộp các text nằm gần nhau thành cụm logic"""
        text_elements = [e for e in elements if e.get('type') == 'text']
        non_text_elements = [e for e in elements if e.get('type') != 'text']

        if len(text_elements) <= 1:
            return text_elements + non_text_elements

        # Sắp xếp các text theo vị trí y (từ trên xuống)
        text_elements.sort(key=lambda e: e['bbox'][1])

        # Gộp theo dòng với ngưỡng linh hoạt hơn
        line_groups = []
        current_line = [text_elements[0]]

        for i in range(1, len(text_elements)):
            curr_elem = text_elements[i]

            # So sánh với tất cả elements trong dòng hiện tại
            min_vertical_distance = float('inf')
            for prev_elem in current_line:
                _, y1, _, h1 = prev_elem['bbox']
                _, y2, _, h2 = curr_elem['bbox']

                # Tính khoảng cách giữa tâm điểm theo chiều dọc
                distance = abs((y1 + h1 / 2) - (y2 + h2 / 2))
                min_vertical_distance = min(min_vertical_distance, distance)

            # Nếu khoảng cách dọc đủ nhỏ -> cùng dòng
            if min_vertical_distance < max(h1, h2) * 0.7:  # Ngưỡng cao hơn (0.7 thay vì 0.5)
                current_line.append(curr_elem)
            else:
                # Khác dòng
                if current_line:
                    line_groups.append(current_line)
                current_line = [curr_elem]

        # Thêm dòng cuối cùng
        if current_line:
            line_groups.append(current_line)

        # Gộp text trong mỗi dòng (sắp xếp từ trái sang phải)
        merged_texts = []

        for line in line_groups:
            # Sắp xếp theo x
            line.sort(key=lambda e: e['bbox'][0])

            if len(line) == 1:
                merged_texts.append(line[0])
            else:
                # Hợp nhất text và bbox
                merged_text = ""
                x_min = min(e['bbox'][0] for e in line)
                y_min = min(e['bbox'][1] for e in line)
                x_max = max(e['bbox'][0] + e['bbox'][2] for e in line)
                y_max = max(e['bbox'][1] + e['bbox'][3] for e in line)

                for i, elem in enumerate(line):
                    if i > 0:
                        # Thêm khoảng trắng chỉ khi cần thiết
                        prev_elem = line[i - 1]
                        prev_end_x = prev_elem['bbox'][0] + prev_elem['bbox'][2]
                        curr_start_x = elem['bbox'][0]

                        # Nếu khoảng cách giữa các từ đủ lớn
                        if curr_start_x - prev_end_x > min(prev_elem['bbox'][2], elem['bbox'][2]) * 0.2:
                            merged_text += " "
                    merged_text += elem.get('text', '')

                # Tạo element mới
                merged_element = {
                    'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                    'text': merged_text,
                    'type': 'text',
                    'merged': True,
                    'confidence': max(e.get('confidence', 0) for e in line),
                    'roi': line[0].get('roi', None),  # Giữ ROI của phần tử đầu tiên
                    'contour': line[0].get('contour', None)  # Giữ contour của phần tử đầu tiên
                }

                merged_texts.append(merged_element)

        return merged_texts + non_text_elements

    def merge_text_elements(self, elements):
        """Gộp các text box gần nhau (mở rộng bbox và kiểm tra giao nhau)."""
        text_elements = [e for e in elements if e.get('type') == 'text']
        non_text_elements = [e for e in elements if e.get('type') != 'text']

        if not text_elements:
            return elements

        # --- (Phần mở rộng bbox và gộp giữ nguyên) ---
        # 1. Mở rộng bbox
        expanded_boxes = []
        for elem in text_elements:
            x, y, w, h = elem['bbox']
            y_top_expansion = int(h * 0.25)  # 25% chiều cao
            y_bottom_expansion = int(h * 0.25)

            new_y = max(0, y - y_top_expansion)  # Đảm bảo không vượt quá biên trên
            new_h = h + y_top_expansion + y_bottom_expansion

            # Cập nhật cả 'bbox' và 'contour' (nếu có)
            new_bbox = (x, new_y, w, new_h)
            new_contour = None
            if 'contour' in elem and elem['contour'] is not None:  # Kiểm tra contour
                # Cập nhật contour (giả sử contour là một numpy array)
                new_contour = elem['contour'].copy()
                new_contour[:, 0, 1] = new_contour[:, 0, 1] - y + new_y

            expanded_boxes.append({
                'bbox': new_bbox,
                'text': elem.get('text', ''),
                'type': 'text',
                'merged': False,  # Ban đầu, chưa box nào được gộp
                'confidence': elem.get('confidence', 0),
                'roi': elem.get('roi', None),
                'contour':new_contour,
                'original_bbox': elem['bbox'], #Lưu lại original
                'original_contour': elem.get('contour',None) #Lưu lại original
            })

        # 2. Kiểm tra giao nhau và gộp
        merged_boxes = []
        while True:
            merged = False  # Đánh dấu xem có cặp nào được gộp trong vòng lặp này không
            used = [False] * len(expanded_boxes)  # Đánh dấu các box đã được gộp

            for i in range(len(expanded_boxes)):
                if used[i]:
                    continue

                current_box = expanded_boxes[i]
                current_group = [current_box]
                used[i] = True

                for j in range(i + 1, len(expanded_boxes)):
                    if used[j]:
                        continue

                    if self.boxes_intersect(current_box['bbox'], expanded_boxes[j]['bbox']):
                        current_group.append(expanded_boxes[j])
                        used[j] = True
                        merged = True  # Đánh dấu đã có gộp

                # Gộp các box trong current_group
                if len(current_group) > 1:
                    merged_text = ""
                    x_min = min(b['bbox'][0] for b in current_group)
                    y_min = min(b['bbox'][1] for b in current_group)
                    x_max = max(b['bbox'][0] + b['bbox'][2] for b in current_group)
                    y_max = max(b['bbox'][1] + b['bbox'][3] for b in current_group)

                    for k, elem in enumerate(current_group):
                        if k > 0:
                            # Thêm khoảng trắng (nếu cần)
                            prev_elem = current_group[k-1]
                            if elem['bbox'][0] > prev_elem["bbox"][0] + prev_elem['bbox'][2]:
                                merged_text += " "
                        merged_text += elem['text']

                    merged_boxes.append({
                        'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                        'text': merged_text,
                        'type': 'text',
                        'merged': True,
                        'confidence': max(e.get('confidence', 0) for e in current_group),
                        'roi': current_group[0].get('roi', None),
                        'contour': current_group[0].get("contour",None)
                    })
                else:
                    merged_boxes.append(current_box)  # Không gộp, thêm vào như cũ


            if not merged:  # Nếu không có cặp nào được gộp, dừng vòng lặp
                break

            expanded_boxes = merged_boxes  # Cập nhật danh sách box cho vòng lặp tiếp theo
            merged_boxes = []
        # --- (Hết phần gộp) ---

        # 3. Khôi phục kích thước ban đầu cho các box KHÔNG bị gộp
        final_boxes = []
        for box in merged_boxes:
            if not box.get('merged', False):  # Nếu box không được đánh dấu là đã gộp
                # Khôi phục bbox và contour ban đầu
                original_bbox = box['original_bbox']
                original_contour = box.get('original_contour', None)  # Lấy original_contour, nếu có
                final_boxes.append({
                    'bbox': original_bbox,
                    'text': box['text'],
                    'type': 'text',
                    'merged': False,
                    'confidence': box['confidence'],
                    'roi': box.get('roi', None),
                    'contour': original_contour  # Khôi phục contour
                })
            else:
                final_boxes.append(box)  # Giữ nguyên box đã gộp

        return final_boxes + non_text_elements
    
    #Hàm check 2 box có intersect không
    def boxes_intersect(self, bbox1, bbox2):
        """Kiểm tra hai bounding box có giao nhau không."""
        x1_a, y1_a, w_a, h_a = bbox1
        x1_b, y1_b, w_b, h_b = bbox2

        intersect_x1 = max(x1_a, x1_b)
        intersect_y1 = max(y1_a, y1_b)
        intersect_x2 = min(x1_a + w_a, x1_b + w_b)
        intersect_y2 = min(y1_a + h_a, y1_b + h_b)

        return intersect_x2 > intersect_x1 and intersect_y2 > intersect_y1