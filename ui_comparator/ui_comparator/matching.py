"""Module so khớp các thành phần UI."""

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

class ElementMatcher:
    """So khớp các thành phần UI."""

    def __init__(self):
        pass

    def compute_position_similarity(self, element1, element2, img1_shape, img2_shape):
        """Tính độ tương đồng về vị trí."""
        x1, y1, w1, h1 = element1['bbox']
        x2, y2, w2, h2 = element2['bbox']

        # Tính tọa độ trung tâm
        center1_x = x1 + w1/2
        center1_y = y1 + h1/2
        center2_x = x2 + w2/2
        center2_y = y2 + h2/2

        # Chuẩn hóa tọa độ
        normalized_center1 = (center1_x / img1_shape[1], center1_y / img1_shape[0])
        normalized_center2 = (center2_x / img2_shape[1], center2_y / img2_shape[0])

        # Tính khoảng cách Euclidean
        distance = np.sqrt((normalized_center1[0] - normalized_center2[0])**2 +
                          (normalized_center1[1] - normalized_center2[1])**2)

        # Chuyển khoảng cách thành độ tương đồng
        similarity = max(0, 1 - distance * 1.5)  # Điều chỉnh hệ số giảm để ít phụ thuộc vào vị trí

        return similarity

    def compute_size_similarity(self, element1, element2, img1_shape, img2_shape):
        """Tính độ tương đồng về kích thước."""
        x1, y1, w1, h1 = element1['bbox']
        x2, y2, w2, h2 = element2['bbox']

        # Chuẩn hóa kích thước
        norm_w1 = w1 / img1_shape[1]
        norm_h1 = h1 / img1_shape[0]
        norm_w2 = w2 / img2_shape[1]
        norm_h2 = h2 / img2_shape[0]

        # Tính độ chênh lệch
        w_diff = abs(norm_w1 - norm_w2)
        h_diff = abs(norm_h1 - norm_h2)

        # Tính độ tương đồng
        size_similarity = max(0, 1 - (w_diff + h_diff))

        return size_similarity

    def compute_appearance_similarity(self, element1, element2):
        """Tính độ tương đồng về hình dạng và màu sắc."""
        roi1 = element1.get('roi', None)
        roi2 = element2.get('roi', None)

        # Bỏ qua nếu không có ROI
        if roi1 is None or roi2 is None:
            return 0.0

        # Đảm bảo ROIs có cùng kích thước
        try:
            h1, w1 = roi1.shape[:2]
            h2, w2 = roi2.shape[:2]

            # Resize về kích thước trung bình
            target_h = max(20, min(h1, h2))
            target_w = max(20, min(w1, w2))

            roi1_resized = cv2.resize(roi1, (target_w, target_h))
            roi2_resized = cv2.resize(roi2, (target_w, target_h))

            # Chuyển sang grayscale
            if len(roi1_resized.shape) == 3:
                gray1 = cv2.cvtColor(roi1_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = roi1_resized

            if len(roi2_resized.shape) == 3:
                gray2 = cv2.cvtColor(roi2_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = roi2_resized

            # Tính histogram
            hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])

            # Chuẩn hóa histogram
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

            # So sánh histogram
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            # Tăng cường so sánh ngoại hình bằng phương pháp SSIM
            try:
                from skimage.metrics import structural_similarity as ssim
                ssim_score = ssim(gray1, gray2, data_range=255)

                # Kết hợp histogram và SSIM
                appearance_sim = 0.6 * max(0, similarity) + 0.4 * ssim_score
                return appearance_sim
            except:
                # Nếu không có skimage hoặc lỗi, chỉ dùng histogram
                return max(0, similarity)

        except Exception as e:
            print(f"Error in appearance comparison: {e}")
            return 0.0

    def compute_type_similarity(self, element1, element2):
        """Tính độ tương đồng dựa trên loại phần tử."""
        type1 = element1.get('type', 'unknown')
        type2 = element2.get('type', 'unknown')

        if type1 == type2:
            return 1.0

        # Độ tương đồng một phần giữa các loại liên quan
        if (type1 in ['text', 'button'] and type2 in ['text', 'button']):
            return 0.7  # Tăng độ tương đồng

        if (type1 in ['icon', 'image'] and type2 in ['icon', 'image']):
            return 0.7  # Tăng độ tương đồng

        return 0.0

    def compute_text_similarity(self, element1, element2):
        """Tính độ tương đồng về nội dung text."""
        text1 = element1.get('text', '')
        text2 = element2.get('text', '')

        if not text1 or not text2:
            return 0.0

        # Chuẩn hóa text
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        if text1 == text2:
            return 1.0

        # Tính độ tương đồng dựa trên số từ giống nhau
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        common_words = words1.intersection(words2)
        jaccard_similarity = len(common_words) / len(words1.union(words2))

        # Tính thêm độ tương đồng dựa trên ký tự chung
        min_len = min(len(text1), len(text2))
        max_len = max(len(text1), len(text2))

        if min_len == 0:
            return 0.0

        # Tìm số ký tự liên tiếp chung dài nhất
        import difflib
        matcher = difflib.SequenceMatcher(None, text1, text2)
        longest_match = matcher.find_longest_match(0, len(text1), 0, len(text2))

        char_similarity = longest_match.size / max_len if max_len > 0 else 0

        # Kết hợp Jaccard và độ tương đồng ký tự
        text_sim = 0.6 * jaccard_similarity + 0.4 * char_similarity

        return text_sim

    def compute_overall_similarity(self, element1, element2, img1_shape, img2_shape):
        """Tính độ tương đồng tổng hợp giữa hai phần tử UI."""
        # Tính các điểm thành phần
        position_sim = self.compute_position_similarity(element1, element2, img1_shape, img2_shape)
        size_sim = self.compute_size_similarity(element1, element2, img1_shape, img2_shape)
        appearance_sim = self.compute_appearance_similarity(element1, element2)
        type_sim = self.compute_type_similarity(element1, element2)
        text_sim = self.compute_text_similarity(element1, element2)

        # Trọng số cho từng loại điểm - Điều chỉnh lại
        weights = {
            'position': 0.2,    # Giảm trọng số vị trí
            'size': 0.15,       # Giữ nguyên trọng số kích thước
            'appearance': 0.25, # Tăng trọng số ngoại hình
            'type': 0.15,       # Giữ nguyên trọng số loại
            'text': 0.25        # Tăng trọng số text
        }

        # Ưu tiên cao hơn cho text nếu cả hai phần tử có text
        if element1.get('text') and element2.get('text'):
            weights['text'] = 0.35
            weights['position'] = 0.15
            weights['appearance'] = 0.2

        # Tính điểm trung bình có trọng số
        overall_sim = (
            weights['position'] * position_sim +
            weights['size'] * size_sim +
            weights['appearance'] * appearance_sim +
            weights['type'] * type_sim +
            weights['text'] * text_sim
        )

        # Debug
        # print(f"Position: {position_sim:.2f}, Size: {size_sim:.2f}, Appearance: {appearance_sim:.2f}, "
        #      f"Type: {type_sim:.2f}, Text: {text_sim:.2f} => Overall: {overall_sim:.2f}")

        return overall_sim

    def match_elements(self, elements1, elements2, img1_shape, img2_shape):
        """Tìm các cặp thành phần tương ứng."""
        n1 = len(elements1)
        n2 = len(elements2)

        if n1 == 0 or n2 == 0:
            return []

        # Tạo ma trận chi phí
        cost_matrix = np.zeros((n1, n2))
        similarity_matrix = np.zeros((n1, n2))  # Để debug

        for i in range(n1):
            for j in range(n2):
                # Tính độ tương đồng tổng hợp
                similarity = self.compute_overall_similarity(
                    elements1[i], elements2[j], img1_shape, img2_shape)

                similarity_matrix[i, j] = similarity  # Để debug

                # Chuyển thành chi phí (càng thấp càng tốt)
                cost_matrix[i, j] = 1.0 - similarity

        # In ma trận tương đồng để debug
        # print("Similarity Matrix:")
        # for i in range(n1):
        #     for j in range(n2):
        #         print(f"{similarity_matrix[i, j]:.2f}", end=" ")
        #     print()

        # Thuật toán Hungarian
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Tạo danh sách cặp ghép
        matches = []
        for i, j in zip(row_ind, col_ind):
            similarity = 1.0 - cost_matrix[i, j]
            if similarity > 0.35:  # Giảm ngưỡng chấp nhận ghép cặp để ghép được nhiều hơn
                # Thêm thông tin debug vào kết quả
                e1_text = elements1[i].get('text', 'No text')
                e2_text = elements2[j].get('text', 'No text')

                # print(f"Matched: {e1_text[:20]} with {e2_text[:20]} (score: {similarity:.2f})")

                matches.append({
                    'element1': elements1[i],
                    'element2': elements2[j],
                    'similarity': similarity
                })

        return matches