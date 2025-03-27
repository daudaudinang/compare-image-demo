# ui_comparator/matching.py

"""Module so khớp các thành phần UI sử dụng DINOv2, Position, Size."""

import numpy as np
import traceback
import sys

# --- Thư viện cần thiết ---
try:
    from transformers import AutoImageProcessor, AutoModel
    import torch
    from PIL import Image
except ImportError:
    print("Lỗi import trong matching.py: Vui lòng cài đặt 'transformers', 'torch', 'pillow'.")
    sys.exit(1)
try:
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Lỗi import trong matching.py: Vui lòng cài đặt 'scipy', 'scikit-learn'.")
    sys.exit(1)
try:
    import cv2 # Cần cho việc chuyển đổi màu ROI
except ImportError:
    print("Lỗi import trong matching.py: Vui lòng cài đặt 'opencv-python'.")
    sys.exit(1)


# --- Class ElementMatcher Mới ---

class ElementMatcher:
    """
    So khớp các thành phần UI giữa hai ảnh sử dụng DINOv2 kết hợp
    với thông tin vị trí và kích thước.
    Sử dụng thuật toán Hungarian để tìm bộ khớp 1-1 tối ưu,
    sau đó lọc kép theo điểm tổng hợp và ngưỡng vị trí tối thiểu.
    """
    def __init__(self, model_name="facebook/dinov2-base"):
        """
        Khởi tạo Matcher và tải model DINOv2.
        Args:
            model_name (str): Tên model DINOv2 trên Hugging Face Hub.
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cpu"
        self._load_model()

    def _load_model(self):
        """Tải model DINOv2 và processor."""
        if self.model is not None:
            print("Model DINOv2 đã được tải.")
            return True
        try:
            print(f"Đang tải model DINOv2 ({self.model_name})...")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval() # Chuyển sang chế độ đánh giá
            print(f"Tải model DINOv2 thành công. Sử dụng thiết bị: {self.device}")
            return True
        except Exception as e:
            print(f"Lỗi khi tải model DINOv2: {e}")
            self.model = None
            self.processor = None
            return False

    def _get_embeddings(self, elements):
        """Trích xuất DINOv2 embeddings cho danh sách các elements."""
        if self.model is None:
            print("Lỗi: Model DINOv2 chưa được tải.")
            return np.array([]), []

        valid_indices=[]
        images_to_process=[]
        for i, element in enumerate(elements):
            roi = element.get('roi')
            # Kiểm tra ROI chặt chẽ
            if roi is not None and isinstance(roi, np.ndarray) and roi.ndim == 3 and roi.shape[0] > 0 and roi.shape[1] > 0:
                try:
                    pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    images_to_process.append(pil_image)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Warning: Lỗi chuyển đổi ROI sang PIL cho element {element.get('id','?')}: {e}")
            # else: print(f"Debug: Bỏ qua ROI của element {element.get('id','?')}")

        if not images_to_process:
            return np.array([]), []

        embeddings_np = np.array([])
        try: # Batch processing
            batch_size = 32 # Giữ batch size hợp lý
            num_batches = (len(images_to_process) + batch_size - 1) // batch_size
            all_features = []
            print(f"    Tạo DINOv2 embeddings cho {len(images_to_process)} ảnh ({num_batches} batches)...")

            for i in range(num_batches):
                batch_images = images_to_process[i*batch_size:(i+1)*batch_size]
                if not batch_images: continue

                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Lấy CLS token embedding hoặc pooler output
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                     image_features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                     image_features = outputs.last_hidden_state[:, 0]
                else:
                     print("Lỗi: Không tìm thấy output embedding phù hợp từ DINOv2.")
                     return np.array([]),[]

                # Chuẩn hóa L2
                image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
                all_features.append(image_features.cpu())

            if all_features:
                embeddings_tensor = torch.cat(all_features, dim=0)
                embeddings_np = embeddings_tensor.numpy()

        except Exception as e:
            print(f"Lỗi trong quá trình xử lý DINOv2: {e}")
            traceback.print_exc()
            return np.array([]), []

        return embeddings_np, valid_indices

    # --- Các hàm tính similarity thành phần (static methods) ---
    @staticmethod
    def _compute_position_similarity(element1, element2, img1_shape, img2_shape):
        """Tính độ tương đồng vị trí chuẩn hóa."""
        # ... (code giữ nguyên như hàm helper trước đó) ...
        bbox1 = element1.get('bbox'); bbox2 = element2.get('bbox');
        if bbox1 is None or bbox2 is None: return 0.0
        try: x1,y1,w1,h1=map(float,bbox1); x2,y2,w2,h2=map(float,bbox2)
        except: return 0.0
        if img1_shape is None or len(img1_shape)<2 or img1_shape[0]<=0 or img1_shape[1]<=0 or \
           img2_shape is None or len(img2_shape)<2 or img2_shape[0]<=0 or img2_shape[1]<=0: return 0.0
        if w1<=0 or h1<=0 or w2<=0 or h2<=0: return 0.0
        c1x,c1y=x1+w1/2,y1+h1/2; c2x,c2y=x2+w2/2,y2+h2/2
        n1=(c1x/img1_shape[1],c1y/img1_shape[0]); n2=(c2x/img2_shape[1],c2y/img2_shape[0])
        dist=np.sqrt((n1[0]-n2[0])**2+(n1[1]-n2[1])**2); return max(0.0,1.0-dist*1.5)

    @staticmethod
    def _compute_size_similarity(element1, element2, img1_shape, img2_shape):
        """Tính độ tương đồng kích thước chuẩn hóa."""
        # ... (code giữ nguyên như hàm helper trước đó) ...
        bbox1 = element1.get('bbox'); bbox2 = element2.get('bbox');
        if bbox1 is None or bbox2 is None: return 0.0
        try: x1,y1,w1,h1=map(float,bbox1); x2,y2,w2,h2=map(float,bbox2)
        except: return 0.0
        if img1_shape is None or len(img1_shape)<2 or img1_shape[0]<=0 or img1_shape[1]<=0 or \
           img2_shape is None or len(img2_shape)<2 or img2_shape[0]<=0 or img2_shape[1]<=0: return 0.0
        if w1<=0 or h1<=0 or w2<=0 or h2<=0: return 0.0
        nw1,nh1=w1/img1_shape[1],h1/img1_shape[0]; nw2,nh2=w2/img2_shape[1],h2/img2_shape[0]
        wd,hd=abs(nw1-nw2),abs(nh1-nh2); return max(0.0,1.0-(wd+hd))


    # --- Phương thức Matching chính ---
    def match_elements(self, elements_design, elements_real, img_design_shape, img_real_shape,
                       weights={'dino': 0.6, 'pos': 0.3, 'size': 0.1}, # Trọng số mặc định
                       combined_threshold=0.55,                       # Ngưỡng tổng hợp mặc định
                       min_pos_threshold=0.4):                        # Ngưỡng vị trí tối thiểu mặc định
        """
        Thực hiện matching giữa hai bộ elements.

        Args:
            elements_design (list): List các dictionary element của ảnh design.
            elements_real (list): List các dictionary element của ảnh real.
            img_design_shape (tuple): Shape (height, width, ...) của ảnh design.
            img_real_shape (tuple): Shape (height, width, ...) của ảnh real.
            weights (dict): Dictionary chứa trọng số cho 'dino', 'pos', 'size'.
            combined_threshold (float): Ngưỡng cuối cùng cho điểm tương đồng tổng hợp.
            min_pos_threshold (float): Ngưỡng vị trí tối thiểu để giữ lại cặp khớp sau Hungarian.

        Returns:
            list: Danh sách các dictionary khớp, mỗi dict chứa 'element1', 'element2',
                  'similarity' (điểm tổng hợp), và các điểm thành phần.
        """
        method_name = f"Matcher(DINOv2+Pos+Size Filtered)" # Tên để debug/log
        print(f"\n--- {method_name}: Bắt đầu matching ---")

        if self.model is None:
            print(f"{method_name}: Model DINOv2 lỗi, không thể matching."); return []
        n1 = len(elements_design); n2 = len(elements_real)
        if n1 == 0 or n2 == 0:
             print(f"{method_name}: Danh sách elements rỗng."); return []
        if img_design_shape is None or img_real_shape is None:
             print(f"{method_name}: Thiếu thông tin shape ảnh."); return []

        # 1. Tính DINOv2 Image Similarity
        design_embeddings, valid_design_indices = self._get_embeddings(elements_design)
        real_embeddings, valid_real_indices = self._get_embeddings(elements_real)
        dinov2_similarity_matrix_full = np.zeros((n1, n2), dtype=np.float64)
        if design_embeddings.size > 0 and real_embeddings.size > 0:
            if design_embeddings.ndim==1: design_embeddings=design_embeddings.reshape(1,-1)
            if real_embeddings.ndim==1: real_embeddings=real_embeddings.reshape(1,-1)
            similarity_matrix_valid = cosine_similarity(design_embeddings, real_embeddings)
            for i_valid, idx1 in enumerate(valid_design_indices):
                for j_valid, idx2 in enumerate(valid_real_indices):
                     if i_valid<similarity_matrix_valid.shape[0] and j_valid<similarity_matrix_valid.shape[1]:
                         dinov2_similarity_matrix_full[idx1, idx2] = np.clip(similarity_matrix_valid[i_valid, j_valid], 0.0, 1.0)
        else:
            print(f"{method_name}: Không tạo được DINOv2 embeddings.")
            # Vẫn có thể tiếp tục nếu chỉ dựa vào Pos/Size, nhưng kết quả sẽ kém

        # 2. Tính Position và Size Similarity
        pos_similarity_matrix_full = np.zeros((n1, n2), dtype=np.float64)
        size_similarity_matrix_full = np.zeros((n1, n2), dtype=np.float64)
        valid_bbox_mask = np.zeros((n1, n2), dtype=bool)
        print(f"{method_name}: Tính Pos/Size similarity...")
        for i in range(n1):
            for j in range(n2):
                 # Chỉ tính nếu có bbox hợp lệ
                 if 'bbox' in elements_design[i] and 'bbox' in elements_real[j] and \
                    isinstance(elements_design[i]['bbox'], (tuple, list)) and len(elements_design[i]['bbox']) == 4 and \
                    isinstance(elements_real[j]['bbox'], (tuple, list)) and len(elements_real[j]['bbox']) == 4:
                     pos_similarity_matrix_full[i, j] = self._compute_position_similarity(elements_design[i], elements_real[j], img_design_shape, img_real_shape)
                     size_similarity_matrix_full[i, j] = self._compute_size_similarity(elements_design[i], elements_real[j], img_design_shape, img_real_shape)
                     valid_bbox_mask[i, j] = True

        # 3. Tính điểm tổng hợp và Cost Matrix
        final_similarity_matrix = np.zeros((n1, n2), dtype=np.float64)
        cost_matrix = np.full((n1, n2), 100.0, dtype=np.float64) # Chi phí RẤT CAO
        print(f"{method_name}: Tính điểm tổng hợp...")
        # Tính điểm tổng hợp chỉ cho các cặp có bbox hợp lệ
        final_similarity_matrix[valid_bbox_mask] = (
            weights['dino'] * dinov2_similarity_matrix_full[valid_bbox_mask] +
            weights['pos']  * pos_similarity_matrix_full[valid_bbox_mask] +
            weights['size'] * size_similarity_matrix_full[valid_bbox_mask]
        )
        # Cập nhật cost, đảm bảo không âm
        cost_matrix[valid_bbox_mask] = np.maximum(1.0 - final_similarity_matrix[valid_bbox_mask], 0.0)

        # 4. Thuật toán Hungarian
        print(f"{method_name}: Chạy Hungarian...")
        try: row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ValueError as e: print(f"Lỗi Hungarian {method_name}: {e}"); return []

        # 5. Lọc kết quả kép
        matches = []
        print(f"{method_name}: Lọc kết quả (Combined > {combined_threshold}, Min Pos > {min_pos_threshold})...")
        for r, c in zip(row_ind, col_ind):
            # Chỉ xét cặp được Hungarian chọn VÀ có cost < 99 (đã được tính toán)
            if r < n1 and c < n2 and cost_matrix[r, c] < 99.0:
                final_sim_score = 1.0 - cost_matrix[r, c]
                pos_sim = pos_similarity_matrix_full[r, c] # Lấy điểm vị trí

                # Áp dụng lọc kép
                if final_sim_score >= combined_threshold and pos_sim >= min_pos_threshold:
                     # Lấy các điểm thành phần khác
                     dino_sim = dinov2_similarity_matrix_full[r, c]
                     size_sim = size_similarity_matrix_full[r, c]
                     # Tạo dict kết quả khớp
                     match_data = {
                         'element1':elements_design[r],
                         'element2':elements_real[c],
                         'similarity':final_sim_score, # Điểm tổng hợp quyết định
                         'dino_similarity':dino_sim,
                         'pos_similarity':pos_sim,
                         'size_similarity':size_sim
                     }
                     # Thêm ID nếu có để dễ debug
                     match_data['element1_id'] = elements_design[r].get('id', f'D{r}')
                     match_data['element2_id'] = elements_real[c].get('id', f'R{c}')
                     matches.append(match_data)

        print(f"{method_name}: Tìm thấy {len(matches)} cặp khớp sau khi lọc.")
        return matches