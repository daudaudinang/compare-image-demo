"""Module tiền xử lý ảnh."""

import cv2
import numpy as np
import math

class Preprocessor:
    """Tiền xử lý ảnh, phát hiện màn hình và biến đổi phối cảnh."""
    
    def __init__(self):
        self.design_img = None
        self.real_img = None
        self.processed_design = None
        self.processed_real = None
        self.screen_contour = None
        self.corners = None
    
    def load_images(self, design_path, real_path):
        """Tải ảnh design và ảnh thực tế."""
        self.design_img = cv2.imread(design_path)
        self.real_img = cv2.imread(real_path)
        
        if self.design_img is None or self.real_img is None:
            raise ValueError("Không thể đọc ảnh!")
            
        return self.design_img, self.real_img
    
    def detect_screen(self, img):
        """Phát hiện màn hình trong ảnh."""
        # Chuyển sang grayscale và làm mờ
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Phát hiện cạnh với Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Tăng cường cạnh
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Tìm contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Lọc contours theo diện tích
        min_area = 0.05 * img.shape[0] * img.shape[1]
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not large_contours:
            print("Không tìm thấy màn hình!")
            return None
        
        # Chọn contour lớn nhất là màn hình
        screen_contour = max(large_contours, key=cv2.contourArea)
        
        return screen_contour
    
    def find_virtual_corners(self, img, screen_contour):
        """Tìm các góc ảo của màn hình."""
        # Đơn giản hóa contour
        epsilon = 0.02 * cv2.arcLength(screen_contour, True)
        approx = cv2.approxPolyDP(screen_contour, epsilon, True)
        
        # Trích xuất các đoạn thẳng
        segments = []
        for i in range(len(approx)):
            p1 = tuple(approx[i][0])
            p2 = tuple(approx[(i+1) % len(approx)][0])
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Tính góc của đoạn thẳng
            angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180
            segments.append((p1, p2, length, angle))
        
        # Phân loại các cạnh thành 4 nhóm theo hướng
        horizontal_segments = []  # Gần ngang
        vertical_segments = []    # Gần dọc
        
        for seg in segments:
            _, _, length, angle = seg
            if length < 20:  # Bỏ qua các đoạn quá ngắn
                continue
                
            if 45 <= angle <= 135:
                vertical_segments.append(seg)
            else:
                horizontal_segments.append(seg)
        
        # Phân loại theo vị trí (top, bottom, left, right)
        img_height, img_width = img.shape[:2]
        top_segments = []
        bottom_segments = []
        left_segments = []
        right_segments = []
        
        # Xác định top/bottom từ các cạnh ngang
        for seg in horizontal_segments:
            p1, p2, _, _ = seg
            y_mid = (p1[1] + p2[1]) / 2
            if y_mid < img_height / 2:
                top_segments.append(seg)
            else:
                bottom_segments.append(seg)
        
        # Xác định left/right từ các cạnh dọc
        for seg in vertical_segments:
            p1, p2, _, _ = seg
            x_mid = (p1[0] + p2[0]) / 2
            if x_mid < img_width / 2:
                left_segments.append(seg)
            else:
                right_segments.append(seg)
        
        # Chọn một đoạn đại diện cho mỗi cạnh của màn hình
        main_segments = []
        
        if top_segments:
            main_segments.append(max(top_segments, key=lambda x: x[2]))  # Cạnh dài nhất
        if bottom_segments:
            main_segments.append(max(bottom_segments, key=lambda x: x[2]))
        if left_segments:
            main_segments.append(max(left_segments, key=lambda x: x[2]))
        if right_segments:
            main_segments.append(max(right_segments, key=lambda x: x[2]))
        
        if len(main_segments) < 4:
            print(f"Không tìm đủ 4 cạnh chính của màn hình (chỉ tìm thấy {len(main_segments)})")
            return None
        
        # Kéo dài các cạnh
        extended_segments = []
        for p1, p2, _, _ in main_segments:
            # Kéo dài đoạn thẳng ra xa
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Chuẩn hóa vector
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx /= length
                dy /= length
            
            # Kéo dài 2000 đơn vị về cả hai phía
            distance = 2000
            ext_p1 = (int(p1[0] - dx * distance), int(p1[1] - dy * distance))
            ext_p2 = (int(p2[0] + dx * distance), int(p2[1] + dy * distance))
            
            extended_segments.append((ext_p1, ext_p2))
        
        # Tính các góc ảo từ giao điểm của các cạnh được kéo dài
        lines = []
        for p1, p2 in extended_segments:
            # Tính hệ số đường thẳng ax + by + c = 0
            a = p2[1] - p1[1]
            b = p1[0] - p2[0]
            c = p2[0]*p1[1] - p1[0]*p2[1]
            
            # Chuẩn hóa
            norm = math.sqrt(a*a + b*b)
            if norm != 0:
                a, b, c = a/norm, b/norm, c/norm
            
            lines.append((a, b, c))
        
        # Tìm các giao điểm
        intersection_points = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                a1, b1, c1 = lines[i]
                a2, b2, c2 = lines[j]
                
                # Kiểm tra hai đường thẳng không song song
                det = a1*b2 - a2*b1
                if abs(det) < 1e-8:
                    continue
                
                # Tính giao điểm
                x = (b1*c2 - b2*c1) / det
                y = (a2*c1 - a1*c2) / det
                
                # Kiểm tra điểm có hợp lý không
                if -5000 <= x <= 5000 and -5000 <= y <= 5000:
                    intersection_points.append((x, y))
        
        if len(intersection_points) < 4:
            print(f"Không tìm đủ 4 giao điểm (chỉ tìm được {len(intersection_points)})")
            return None
        
        # Lọc 4 giao điểm tạo thành hình tứ giác
        corners = self.filter_corners(intersection_points)
        if len(corners) != 4:
            print(f"Không thể lọc ra 4 góc (có {len(corners)} điểm)")
            return None
        
        # Sắp xếp góc theo thứ tự: top-left, top-right, bottom-right, bottom-left
        corners = self.order_points(np.array(corners, dtype=np.float32))
        
        return corners
    
    def filter_corners(self, points):
        """Lọc ra 4 góc tạo thành hình tứ giác hợp lý."""
        if len(points) <= 4:
            return points
        
        # Tính centroid của các điểm
        centroid_x = sum(p[0] for p in points) / len(points)
        centroid_y = sum(p[1] for p in points) / len(points)
        
        # Sắp xếp các điểm theo góc so với centroid
        angles = []
        for point in points:
            angle = math.atan2(point[1] - centroid_y, point[0] - centroid_x)
            angles.append((angle, point))
        
        # Sắp xếp theo góc
        angles.sort()
        
        # Chọn 4 điểm ở 4 góc phần tư
        quadrants = [[], [], [], []]  # Phân chia làm 4 góc phần tư
        for angle, point in angles:
            # Xác định góc phần tư (0-3)
            quadrant = int((angle + math.pi) / (math.pi/2)) % 4
            quadrants[quadrant].append(point)
        
        # Chọn điểm xa nhất từ tâm trong mỗi góc phần tư
        final_corners = []
        for quadrant_points in quadrants:
            if quadrant_points:
                # Tính điểm xa nhất từ tâm
                furthest_point = max(quadrant_points, 
                                   key=lambda p: (p[0]-centroid_x)**2 + (p[1]-centroid_y)**2)
                final_corners.append(furthest_point)
        
        return final_corners
    
    def order_points(self, pts):
        """Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left."""
        # Tạo mảng kết quả
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Tìm 2 điểm trên và 2 điểm dưới dựa trên tọa độ y
        # Sắp xếp các điểm theo y
        sorted_by_y = pts[np.argsort(pts[:, 1])]
        top_points = sorted_by_y[:2]  # 2 điểm có y nhỏ nhất
        bottom_points = sorted_by_y[2:]  # 2 điểm có y lớn nhất
        
        # Sắp xếp 2 điểm trên theo x
        top_points = top_points[np.argsort(top_points[:, 0])]
        rect[0] = top_points[0]  # top-left (x nhỏ nhất, y nhỏ)
        rect[1] = top_points[1]  # top-right (x lớn hơn, y nhỏ)
        
        # Sắp xếp 2 điểm dưới theo x
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
        rect[3] = bottom_points[0]  # bottom-left (x nhỏ, y lớn)
        rect[2] = bottom_points[1]  # bottom-right (x lớn, y lớn)
        
        return rect
    
    def apply_perspective_transform(self, img, corners):
        """Áp dụng biến đổi phối cảnh."""
        # Xác định kích thước ảnh đích
        width_top = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) + 
                          ((corners[1][1] - corners[0][1]) ** 2))
        width_bottom = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) + 
                             ((corners[2][1] - corners[3][1]) ** 2))
        width = max(int(width_top), int(width_bottom))
        
        height_left = np.sqrt(((corners[3][0] - corners[0][0]) ** 2) + 
                            ((corners[3][1] - corners[0][1]) ** 2))
        height_right = np.sqrt(((corners[2][0] - corners[1][0]) ** 2) + 
                             ((corners[2][1] - corners[1][1]) ** 2))
        height = max(int(height_left), int(height_right))
        
        # Phát hiện chế độ dọc/ngang
        is_portrait = height > width
        
        # Định nghĩa 4 góc của ảnh đích
        dst_corners = np.array([
            [0, 0],  # top-left
            [width - 1, 0],  # top-right
            [width - 1, height - 1],  # bottom-right
            [0, height - 1]  # bottom-left
        ], dtype=np.float32)
        
        # Tính ma trận chuyển đổi phối cảnh
        M = cv2.getPerspectiveTransform(corners, dst_corners)
        
        # Áp dụng biến đổi phối cảnh
        warped_img = cv2.warpPerspective(img, M, (width, height))
        
        return warped_img, is_portrait
    
    def process(self, design_path, real_path):
        """Thực hiện toàn bộ quy trình tiền xử lý."""
        self.load_images(design_path, real_path)
        
        # Xử lý ảnh thực tế
        if self.is_screenshot(self.real_img):
            # Nếu là screenshot, không cần biến đổi phối cảnh
            self.processed_real = self.real_img.copy()
            is_portrait = self.processed_real.shape[0] > self.processed_real.shape[1]
        else:
            # Phát hiện màn hình
            self.screen_contour = self.detect_screen(self.real_img)
            
            if self.screen_contour is not None:
                # Tìm các góc ảo
                self.corners = self.find_virtual_corners(self.real_img, self.screen_contour)
                
                if self.corners is not None:
                    # Áp dụng biến đổi phối cảnh
                    self.processed_real, is_portrait = self.apply_perspective_transform(
                        self.real_img, self.corners)
                else:
                    # Nếu không tìm được góc, sử dụng ảnh gốc
                    self.processed_real = self.real_img.copy()
                    is_portrait = self.processed_real.shape[0] > self.processed_real.shape[1]
            else:
                # Nếu không tìm được màn hình, sử dụng ảnh gốc
                self.processed_real = self.real_img.copy()
                is_portrait = self.processed_real.shape[0] > self.processed_real.shape[1]
        
        # Xử lý ảnh design
        self.processed_design = self.design_img.copy()
        
        # Đảm bảo hai ảnh có cùng hướng (dọc/ngang)
        design_is_portrait = self.processed_design.shape[0] > self.processed_design.shape[1]
        
        if is_portrait != design_is_portrait:
            # Xoay ảnh design nếu cần thiết
            self.processed_design = cv2.rotate(self.processed_design, cv2.ROTATE_90_CLOCKWISE)
        
        # Resize ảnh thực tế về cùng kích thước với design
        self.processed_real = cv2.resize(self.processed_real, 
                                       (self.processed_design.shape[1], self.processed_design.shape[0]))
        
        return self.processed_design, self.processed_real
    
    def is_screenshot(self, img):
        """Kiểm tra xem ảnh có phải là screenshot không."""
        # Screenshot thường có các cạnh thẳng và sắc nét
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Đếm số pixel cạnh
        edge_count = np.count_nonzero(edges)
        
        # Nếu số pixel cạnh chiếm tỷ lệ nhỏ và phân bố đều, có thể là screenshot
        h, w = img.shape[:2]
        edge_ratio = edge_count / (h * w)
        
        # Kiểm tra phân bố
        rows = np.sum(edges, axis=1)
        cols = np.sum(edges, axis=0)
        
        row_var = np.var(rows)
        col_var = np.var(cols)
        
        # Screenshot thường có phương sai thấp (phân bố đều)
        # và tỷ lệ cạnh thấp
        return edge_ratio < 0.05 and row_var < 1000 and col_var < 1000
