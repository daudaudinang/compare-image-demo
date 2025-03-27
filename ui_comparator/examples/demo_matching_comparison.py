# examples/demo_matching_comparison.py

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import traceback

# --- Thư viện ---
try:
    # Dùng AutoModel để linh hoạt hơn
    from transformers import AutoImageProcessor, AutoModel
    import torch
    from PIL import Image
except ImportError: print("Lỗi import: transformers, torch, pillow"); sys.exit(1)
# ... (các import khác: skimage, scipy, sklearn, tabulate giữ nguyên) ...
try: from skimage.metrics import structural_similarity as ssim
except ImportError: print("Lỗi import: scikit-image"); sys.exit(1)
try: from scipy.optimize import linear_sum_assignment
except ImportError: print("Lỗi import: scipy"); sys.exit(1)
try: from sklearn.metrics.pairwise import cosine_similarity
except ImportError: print("Lỗi import: scikit-learn"); sys.exit(1)
try: from tabulate import tabulate
except ImportError: print("Lỗi import: tabulate"); sys.exit(1)


# --- Import từ repo ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from ui_comparator.preprocessing import Preprocessor
    from ui_comparator.segmentation2 import UISegmenter
    from ui_comparator.matching import ElementMatcher # PP1
except ImportError as import_err: print(f"Lỗi import module: {import_err}"); sys.exit(1)

# --- Khởi tạo Models ---
dinov2_model = None
dinov2_processor = None
device = "cpu"

def load_models(): # Đổi tên hàm và chỉ tải DINOv2
    """Tải model DINOv2 và processor."""
    global dinov2_model, dinov2_processor, device
    if dinov2_model is not None:
        print("Model DINOv2 đã được tải.")
        return True
    try:
        print("Đang tải model DINOv2 (facebook/dinov2-base)...")
        # Sử dụng AutoModel để tải
        model_name = "facebook/dinov2-base"
        dinov2_processor = AutoImageProcessor.from_pretrained(model_name)
        dinov2_model = AutoModel.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dinov2_model.to(device)
        dinov2_model.eval() # Chuyển sang chế độ đánh giá
        print(f"Tải model DINOv2 thành công. Sử dụng thiết bị: {device}")
        return True
    except Exception as e:
        print(f"Lỗi khi tải model DINOv2: {e}")
        dinov2_model = None
        dinov2_processor = None
        return False

# --- Các hàm Helper Similarity ---
def compute_position_similarity(element1, element2, img1_shape, img2_shape):
    # ... (code giữ nguyên) ...
    bbox1=element1.get('bbox'); bbox2=element2.get('bbox');
    if bbox1 is None or bbox2 is None: return 0.0
    try: x1,y1,w1,h1=map(float,bbox1); x2,y2,w2,h2=map(float,bbox2)
    except: return 0.0
    if img1_shape is None or len(img1_shape)<2 or img1_shape[0]<=0 or img1_shape[1]<=0 or \
       img2_shape is None or len(img2_shape)<2 or img2_shape[0]<=0 or img2_shape[1]<=0: return 0.0
    if w1<=0 or h1<=0 or w2<=0 or h2<=0: return 0.0
    c1x,c1y=x1+w1/2,y1+h1/2; c2x,c2y=x2+w2/2,y2+h2/2
    n1=(c1x/img1_shape[1],c1y/img1_shape[0]); n2=(c2x/img2_shape[1],c2y/img2_shape[0])
    dist=np.sqrt((n1[0]-n2[0])**2+(n1[1]-n2[1])**2); return max(0.0,1.0-dist*1.5)

def compute_size_similarity(element1, element2, img1_shape, img2_shape):
    # ... (code giữ nguyên) ...
    bbox1=element1.get('bbox'); bbox2=element2.get('bbox');
    if bbox1 is None or bbox2 is None: return 0.0
    try: x1,y1,w1,h1=map(float,bbox1); x2,y2,w2,h2=map(float,bbox2)
    except: return 0.0
    if img1_shape is None or len(img1_shape)<2 or img1_shape[0]<=0 or img1_shape[1]<=0 or \
       img2_shape is None or len(img2_shape)<2 or img2_shape[0]<=0 or img2_shape[1]<=0: return 0.0
    if w1<=0 or h1<=0 or w2<=0 or h2<=0: return 0.0
    nw1,nh1=w1/img1_shape[1],h1/img1_shape[0]; nw2,nh2=w2/img2_shape[1],h2/img2_shape[0]
    wd,hd=abs(nw1-nw2),abs(nh1-nh2); return max(0.0,1.0-(wd+hd))

# --- Các hàm Matching ---

# PP1: ElementMatcher (Giữ nguyên)
def match_focused_similarity(elements_design, elements_real, img_design_shape, img_real_shape):
    # ... (code giữ nguyên) ...
    print("\n--- Thực hiện PP1: Focused Similarity Search (ElementMatcher) ---")
    matcher = ElementMatcher(); matches_raw = []
    try: matches_raw = matcher.match_elements(elements_design, elements_real, img_design_shape, img_real_shape)
    except Exception as e: print(f"Lỗi PP1: {e}"); traceback.print_exc()
    matches = [{'element1': m.get('element1'), 'element2': m.get('element2'), 'similarity': m.get('similarity', 0.0)} for m in matches_raw]
    print(f"PP1: Tìm thấy {len(matches)} cặp khớp."); return matches

# PP2, PP3 (SSIM) có thể bỏ hoặc giữ lại nếu muốn so sánh
# def match_clip_similarity(...) # Bỏ hàm này nếu không dùng CLIP nữa
# def match_ssim(...) # Bỏ hàm này nếu không dùng SSIM nữa

# --- PHƯƠNG PHÁP CHÍNH (PP4 đổi tên thành PP_DINO): DINOv2 + Pos + Size + Hungarian (Weighted Sum + Min Pos Filter) ---
def match_dinov2_pos_size_combined_filtered(
        elements_design, elements_real, img_design_shape, img_real_shape,
        weights={'dino': 0.6, 'pos': 0.3, 'size': 0.1}, # Trọng số KẾT HỢP
        combined_threshold=0.7,                       # Ngưỡng cho điểm TỔNG HỢP
        min_pos_threshold=0.4                         # Ngưỡng VỊ TRÍ TỐI THIỂU sau Hungarian
    ):
    """Tính điểm tổng hợp (DINOv2, Pos, Size), chạy Hungarian, lọc bằng ngưỡng vị trí tối thiểu."""
    method_name = f"PP_DINO: Weighted Sum(dino:{weights['dino']},pos:{weights['pos']},size:{weights['size']}) > {combined_threshold} + Min Pos Filter (>{min_pos_threshold})"
    print(f"\n--- Thực hiện {method_name} ---")

    if dinov2_model is None: print("Model DINOv2 lỗi, bỏ qua."); return []
    n1=len(elements_design); n2=len(elements_real)
    if n1==0 or n2==0: return []

    # 1. Tính DINOv2 Image Similarity
    # --- Hàm con lấy DINOv2 embedding ---
    def get_dinov2_embeddings(elements):
        valid_indices=[]; images_to_process=[]
        for i,element in enumerate(elements):
            roi=element.get('roi')
            if roi is not None and isinstance(roi, np.ndarray) and roi.ndim==3 and roi.shape[0]>0 and roi.shape[1]>0:
                try:
                    # DINOv2 processor thường cần ảnh PIL hoặc Tensor
                    pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    images_to_process.append(pil_image)
                    valid_indices.append(i)
                except Exception as e: print(f"Warn: Lỗi cvt ROI {i}: {e}")
        if not images_to_process: return np.array([]),[]

        embeddings_np=np.array([])
        try: # Batch processing
            batch_size=32 # Thử batch size nhỏ hơn cho DINOv2
            num_batches=(len(images_to_process)+batch_size-1)//batch_size
            all_features=[]
            print(f"    Tạo DINOv2 embeddings cho {len(images_to_process)} ảnh ({num_batches} batches)...")
            for i in range(num_batches):
                batch_images=images_to_process[i*batch_size:(i+1)*batch_size];
                if not batch_images: continue
                # Xử lý ảnh bằng processor của DINOv2
                inputs = dinov2_processor(images=batch_images, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    outputs = dinov2_model(**inputs)
                # Lấy CLS token embedding từ last_hidden_state hoặc pooler_output
                # Kiểm tra cấu trúc output của model cụ thể
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                     image_features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                     image_features = outputs.last_hidden_state[:, 0] # Lấy embedding của token [CLS]
                else:
                     print("Lỗi: Không tìm thấy output embedding phù hợp từ DINOv2.")
                     return np.array([]),[]

                # Chuẩn hóa L2
                image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
                all_features.append(image_features.cpu())

            if all_features:
                embeddings_tensor=torch.cat(all_features,dim=0)
                embeddings_np=embeddings_tensor.numpy()
        except Exception as e:
            print(f"Lỗi trong quá trình xử lý DINOv2: {e}")
            traceback.print_exc()
            return np.array([]),[]
        return embeddings_np, valid_indices
    # --- Hết hàm con ---

    design_embeddings, valid_design_indices = get_dinov2_embeddings(elements_design)
    real_embeddings, valid_real_indices = get_dinov2_embeddings(elements_real)

    dinov2_similarity_matrix_full = np.zeros((n1, n2), dtype=np.float64)
    if design_embeddings.size > 0 and real_embeddings.size > 0:
        if design_embeddings.ndim==1: design_embeddings=design_embeddings.reshape(1,-1)
        if real_embeddings.ndim==1: real_embeddings=real_embeddings.reshape(1,-1)
        similarity_matrix_valid = cosine_similarity(design_embeddings,real_embeddings)
        for i_valid,idx1 in enumerate(valid_design_indices):
            for j_valid,idx2 in enumerate(valid_real_indices):
                 if i_valid<similarity_matrix_valid.shape[0] and j_valid<similarity_matrix_valid.shape[1]:
                     dinov2_similarity_matrix_full[idx1, idx2] = np.clip(similarity_matrix_valid[i_valid, j_valid], 0.0, 1.0)
    else:
         print(f"{method_name}: Ko có embedding DINOv2 hợp lệ.")
         # Vẫn tiếp tục để tính Pos/Size nhưng kết quả cuối sẽ rỗng

    # 2. Tính Position và Size Similarity
    pos_similarity_matrix_full = np.zeros((n1, n2), dtype=np.float64)
    size_similarity_matrix_full = np.zeros((n1, n2), dtype=np.float64)
    valid_bbox_mask = np.zeros((n1, n2), dtype=bool)
    print(f"{method_name}: Tính Pos/Size similarity...")
    for i in range(n1):
        for j in range(n2):
            if 'bbox' in elements_design[i] and 'bbox' in elements_real[j] and \
               isinstance(elements_design[i]['bbox'], (tuple, list)) and len(elements_design[i]['bbox']) == 4 and \
               isinstance(elements_real[j]['bbox'], (tuple, list)) and len(elements_real[j]['bbox']) == 4:
                 pos_similarity_matrix_full[i, j] = compute_position_similarity(elements_design[i], elements_real[j], img_design_shape, img_real_shape)
                 size_similarity_matrix_full[i, j] = compute_size_similarity(elements_design[i], elements_real[j], img_design_shape, img_real_shape)
                 valid_bbox_mask[i, j] = True

    # 3. Tính điểm tổng hợp và Cost Matrix (chỉ cho cặp có bbox hợp lệ)
    final_similarity_matrix = np.zeros((n1, n2), dtype=np.float64)
    cost_matrix = np.full((n1, n2), 100.0, dtype=np.float64) # Chi phí RẤT CAO

    # Tính điểm tổng hợp chỉ cho các cặp có bbox hợp lệ
    final_similarity_matrix[valid_bbox_mask] = (
        weights['dino'] * dinov2_similarity_matrix_full[valid_bbox_mask] + # Dùng điểm DINOv2
        weights['pos']  * pos_similarity_matrix_full[valid_bbox_mask] +
        weights['size'] * size_similarity_matrix_full[valid_bbox_mask]
    )
    # Cập nhật cost, đảm bảo không âm
    cost_matrix[valid_bbox_mask] = np.maximum(1.0 - final_similarity_matrix[valid_bbox_mask], 0.0)

    # 4. Thuật toán Hungarian
    print(f"{method_name}: Chạy Hungarian...")
    try: row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError as e: print(f"Lỗi Hungarian {method_name}: {e}"); return []

    # 5. Lọc kết quả kép: theo ngưỡng tổng hợp VÀ ngưỡng vị trí tối thiểu
    matches = []
    print(f"{method_name}: Lọc kết quả...")
    for r, c in zip(row_ind, col_ind):
        # Chỉ xét cặp được Hungarian chọn VÀ có cost < 99 (đã được tính toán)
        if r < n1 and c < n2 and cost_matrix[r, c] < 99.0:
            final_sim_score = 1.0 - cost_matrix[r, c]
            pos_sim = pos_similarity_matrix_full[r, c] # Lấy điểm vị trí

            # Áp dụng lọc kép
            if final_sim_score >= combined_threshold and pos_sim >= min_pos_threshold:
                 # Lấy các điểm thành phần khác để hiển thị
                 dino_sim = dinov2_similarity_matrix_full[r, c]
                 size_sim = size_similarity_matrix_full[r, c]
                 matches.append({'element1':elements_design[r],'element2':elements_real[c],
                                 'similarity':final_sim_score, 'dino_similarity':dino_sim, # Đổi tên để rõ ràng
                                 'pos_similarity':pos_sim, 'size_similarity':size_sim})

    print(f"{method_name}: Tìm thấy {len(matches)} cặp khớp sau khi lọc.")
    return matches

# --- Hàm Visualize ---
# visualize_segmentation_with_ids (Giữ nguyên)
def visualize_segmentation_with_ids(img, elements, title, ax):
    # ... (code giữ nguyên) ...
    img_vis = img.copy();
    if elements is None: elements = []
    colors={}; np.random.seed(0)
    unique_ids=sorted(list(set(el.get('id',i) for i, el in enumerate(elements))))
    for i,elem_id in enumerate(unique_ids): colors[elem_id]=tuple(np.random.randint(60,256,size=3).tolist())
    for i, element in enumerate(elements):
        element_id=element.get('id', f'NoID_{i}'); bbox=element.get('bbox')
        if bbox is None or not isinstance(bbox,(tuple,list)) or len(bbox)!=4: continue
        try: x,y,w,h=map(int,bbox)
        except: continue
        if w<=0 or h<=0 or x<0 or y<0: continue
        color=colors.get(element_id,(0,0,255))
        try:
            cv2.rectangle(img_vis,(x,y),(x+w,y+h),color,2)
            text_pos=(x+2,y+12) if y+12<y+h else (x+2,y-5)
            cv2.putText(img_vis,str(element_id),text_pos,cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1,cv2.LINE_AA)
        except Exception as e: print(f"Lỗi vẽ box/text ID {element_id}: {e}")
    ax.imshow(cv2.cvtColor(img_vis,cv2.COLOR_BGR2RGB)); ax.set_title(title+f"\n({len(elements)} elements)"); ax.axis('off')

# visualize_matches_lines (Giữ nguyên)
def visualize_matches_lines(img_design, img_real, matches, title, ax):
    # ... (code giữ nguyên) ...
    if img_design is None or img_real is None: ax.text(0.5,0.5,"Lỗi ảnh",ha='center',va='center',transform=ax.transAxes,color='red'); ax.set_title(title+"\n(Lỗi tải ảnh)"); ax.axis('off'); return
    img_design_vis=img_design.copy(); img_real_vis=img_real.copy(); h1,w1=img_design_vis.shape[:2]; h2,w2=img_real_vis.shape[:2]
    np.random.seed(42); colors=[tuple(np.random.randint(50,256,size=3).tolist()) for _ in range(len(matches))]
    valid_matches_count = 0
    for i, match in enumerate(matches):
        if 'element1' not in match or 'element2' not in match or 'similarity' not in match: continue
        e1=match['element1']; e2=match['element2']; sim=match['similarity']
        if 'bbox' not in e1 or 'bbox' not in e2: continue
        bbox1=e1['bbox']; bbox2=e2['bbox']
        if not isinstance(bbox1,(tuple,list)) or len(bbox1)!=4 or not isinstance(bbox2,(tuple,list)) or len(bbox2)!=4: continue
        try: x1,y1,w1b,h1b=map(int,bbox1); x2,y2,w2b,h2b=map(int,bbox2)
        except: continue
        if w1b<=0 or h1b<=0 or w2b<=0 or h2b<=0 or x1<0 or y1<0 or x2<0 or y2<0: continue
        color=colors[i%len(colors)]
        id1=e1.get('id','?'); id2=e2.get('id','?')
        try:
            cv2.rectangle(img_design_vis,(x1,y1),(x1+w1b,y1+h1b),color,2); cv2.putText(img_design_vis,f"{id1}({sim:.2f})",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
            cv2.rectangle(img_real_vis,(x2,y2),(x2+w2b,y2+h2b),color,2); cv2.putText(img_real_vis,f"{id2}({sim:.2f})",(x2,y2-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
            valid_matches_count+=1
        except Exception as draw_error: print(f"Lỗi vẽ box/text nối {i}: {draw_error}"); continue
    vh=max(h1,h2); vw=w1+w2
    if vw<=0: ax.text(0.5,0.5,"Lỗi KT ảnh",ha='center',va='center',transform=ax.transAxes,color='red'); ax.set_title(title+"\n(Lỗi kích thước)"); ax.axis('off'); return
    combined_vis=np.zeros((vh,vw,3),dtype=np.uint8); combined_vis[:h1,:w1,:]=img_design_vis; combined_vis[:h2,w1:w1+w2,:]=img_real_vis
    for i, match in enumerate(matches):
        if 'element1' not in match or 'element2' not in match: continue
        e1=match['element1']; e2=match['element2']
        if 'bbox' not in e1 or 'bbox' not in e2: continue
        bbox1=e1['bbox']; bbox2=e2['bbox']
        if not isinstance(bbox1,(tuple,list)) or len(bbox1)!=4 or not isinstance(bbox2,(tuple,list)) or len(bbox2)!=4: continue
        try:
            x1,y1,w1b,h1b=map(int,bbox1); x2,y2,w2b,h2b=map(int,bbox2)
            if w1b<=0 or h1b<=0 or w2b<=0 or h2b<=0 or x1<0 or y1<0 or x2<0 or y2<0: continue
            c1=(int(x1+w1b/2),int(y1+h1b/2)); c2=(int(w1+x2+w2b/2),int(y2+h2b/2))
            cv2.line(combined_vis,c1,c2,colors[i%len(colors)],1)
        except Exception: continue
    ax.imshow(cv2.cvtColor(combined_vis,cv2.COLOR_BGR2RGB)); ax.set_title(title+f"\n(Hiển thị {valid_matches_count}/{len(matches)} cặp nối)"); ax.axis('off')


# --- Hàm Hiển thị Bảng ---
def display_match_table(matches, method_name="Matching Results"):
    # ... (code giữ nguyên) ...
    print(f"\n--- Bảng Kết quả cho: {method_name} ---")
    if not matches: print("Không tìm thấy cặp khớp nào."); return
    headers = ["Design ID", "Real ID", "Final Sim"]
    component_keys = []
    if matches:
        first_match = matches[0]
        # Lấy tất cả các key similarity khác (ví dụ: dino_similarity, pos_similarity...)
        component_keys = sorted([k for k in first_match.keys() if k.endswith('_similarity') and k != 'similarity'])
        headers.extend([key.replace('_', ' ').title() for key in component_keys])
    table_data = []
    for match in matches:
        row = [ match.get('element1',{}).get('id','N/A'), match.get('element2',{}).get('id','N/A'), f"{match.get('similarity', 0):.3f}" ]
        for key in component_keys: row.append(f"{match.get(key, 0):.3f}")
        table_data.append(row)
    try: table_data.sort(key=lambda row: int(row[0][1:]) if len(row[0]) > 1 and row[0][0] in ('D','R') and row[0][1:].isdigit() else float('inf'))
    except: table_data.sort(key=lambda row: row[0])
    print(tabulate(table_data, headers=headers, tablefmt="grid")); print(f"Tổng cộng: {len(matches)} cặp khớp.")


# --- Hàm Main ---
def main():
    # --- Đặt tên file ảnh ---
    DESIGN_FILENAME = 'design2.jpg'
    REAL_FILENAME = 'real2.jpg'
    # -----------------------

    # --- Ngưỡng và Trọng số ---
    # --- Tham số cho PP_DINO (Weighted Sum + Min Pos Filter) ---
    PP_DINO_WEIGHTS = {'dino': 0.6, 'pos': 0.3, 'size': 0.1} # << Trọng số kết hợp DINO+Pos+Size
    PP_DINO_COMBINED_THRESHOLD = 0.6                    # << Ngưỡng cho điểm TỔNG HỢP
    PP_DINO_MIN_POS_THRESHOLD = 0.4                     # << Ngưỡng VỊ TRÍ TỐI THIỂU sau Hungarian
    # --------------------------------------------------------
    # Các ngưỡng khác (nếu muốn chạy PP1, PP2, PP3 để so sánh)
    # CLIP_PURE_THRESHOLD = 0.7
    # SSIM_THRESHOLD = 0.5
    # SSIM_RESIZE_DIM = (64, 64)
    # --------------------------

    print("="*20 + " BẮT ĐẦU DEMO " + "="*20)
    use_dino = load_models() # Hàm này giờ load DINOv2

    # --- Bước 1 & 2 ---
    print("\nBước 1 & 2: Tải, Tiền xử lý, Phân đoạn, Gán ID & ROI...")
    # ... (Code giữ nguyên) ...
    base_dir=os.path.dirname(__file__); img_folder=os.path.join(base_dir,'sample_images'); design_path=os.path.join(img_folder,DESIGN_FILENAME); real_path=os.path.join(img_folder,REAL_FILENAME)
    if not os.path.exists(design_path): print(f"Lỗi: Ko tìm thấy Design: {design_path}"); return
    if not os.path.exists(real_path): print(f"Lỗi: Ko tìm thấy Real: {real_path}"); return
    img_design_orig=cv2.imread(design_path); img_real_orig=cv2.imread(real_path)
    if img_design_orig is None or img_real_orig is None: print("Lỗi đọc ảnh."); return
    preprocessor=Preprocessor(); img_design=None; img_real=None
    try: img_design,img_real=preprocessor.process(design_path,real_path); print("Tiền xử lý xong.")
    except Exception as e:
        print(f"Lỗi tiền xử lý: {e}. Dùng ảnh gốc."); img_design=img_design_orig.copy(); img_real=img_real_orig.copy()
        target_shape=(img_design.shape[1],img_design.shape[0])
        if img_real.shape[0]!=target_shape[1] or img_real.shape[1]!=target_shape[0]:
             try: img_real=cv2.resize(img_real,target_shape); print(f"Real resized to {target_shape}.")
             except cv2.error as re: print(f"Lỗi resize Real: {re}"); return
    if img_design is None or img_real is None: print("Lỗi: Ảnh None sau tiền xử lý/resize."); return
    if img_design.shape[0]!=img_real.shape[0] or img_design.shape[1]!=img_real.shape[1]: print(f"Lỗi: KT ảnh ko khớp! D:{img_design.shape}, R:{img_real.shape}"); return
    print(f"KT ảnh sau tiền xử lý: D{img_design.shape}, R{img_real.shape}")
    segmenter=UISegmenter()
    try: elements_design=segmenter.segment(img_design); elements_real=segmenter.segment(img_real); print(f"Phân đoạn xong: D({len(elements_design)}), R({len(elements_real)}).")
    except Exception as se: print(f"Lỗi phân đoạn: {se}"); traceback.print_exc(); return
    def assign_ids_and_rois(elements, img, prefix):
        count_valid_roi = 0
        for i, elem in enumerate(elements):
            elem['id'] = f"{prefix}{i}"
            if 'bbox' not in elem or not isinstance(elem['bbox'],(tuple,list)) or len(elem['bbox'])!=4: elem['roi']=None; continue
            try: x,y,w,h=map(int,elem['bbox'])
            except: elem['roi']=None; continue
            if w<=0 or h<=0: elem['roi']=None; continue
            ys=max(0,y); xs=max(0,x)
            if ys>=img.shape[0] or xs>=img.shape[1]: elem['roi']=None; continue
            hs=min(h,img.shape[0]-ys); ws=min(w,img.shape[1]-xs)
            if hs>0 and ws>0: elem['roi']=img[ys:ys+hs,xs:xs+ws].copy(); count_valid_roi+=1
            else: elem['roi']=None
        print(f"Gán ID và {count_valid_roi} ROI cho {len(elements)} elements ({prefix}).")
        return elements
    elements_design = assign_ids_and_rois(elements_design, img_design, 'D')
    elements_real = assign_ids_and_rois(elements_real, img_real, 'R')
    img_design_shape = img_design.shape; img_real_shape = img_real.shape

    # --- Bước 3: Thực hiện Matching ---
    print("\nBước 3: Thực hiện Matching...")
    matches_focused = []; matches_dino_combined = [] # Chỉ chạy PP1 và PP_DINO

    # Chạy PP_DINO (phương pháp chính)
    if use_dino:
        try:
            matches_dino_combined = match_dinov2_pos_size_combined_filtered( # Gọi hàm DINOv2
                elements_design, elements_real, img_design_shape, img_real_shape,
                weights=PP_DINO_WEIGHTS,
                combined_threshold=PP_DINO_COMBINED_THRESHOLD,
                min_pos_threshold=PP_DINO_MIN_POS_THRESHOLD
            )
        except Exception as e: print(f"Lỗi PP_DINO: {e}"); traceback.print_exc()
    else: print("DINOv2 lỗi, không chạy PP_DINO.")

    # Chạy PP1 để so sánh
    try: matches_focused = match_focused_similarity(elements_design, elements_real, img_design_shape, img_real_shape)
    except Exception as e: print(f"Lỗi PP1: {e}"); traceback.print_exc()

    # --- Bước 4: Hiển thị Bảng và Visualize ---
    print("\nBước 4: Hiển thị Bảng Kết quả và Visualize...")

    # --- Hiển thị bảng KẾT QUẢ PP_DINO ---
    display_match_table(matches_dino_combined, method_name=f"PP_DINO: Weighted Sum + Min Pos Filter (>{PP_DINO_MIN_POS_THRESHOLD})")

    # --- Visualize Ảnh ---
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    axes = axes.ravel()

    # Vẽ ảnh với ID (axes[0], axes[1])
    visualize_segmentation_with_ids(img_design, elements_design, f"Design: {DESIGN_FILENAME}", axes[0])
    visualize_segmentation_with_ids(img_real, elements_real, f"Real: {REAL_FILENAME}", axes[1])

    # Vẽ kết quả nối của PP_DINO (axes[2])
    if use_dino:
        visualize_matches_lines(img_design, img_real, matches_dino_combined, f"PP_DINO: Weighted Sum + Min Pos Filter(>{PP_DINO_MIN_POS_THRESHOLD})", axes[2])
    else:
        axes[2].text(0.5,0.5,"DINOv2 Model lỗi",ha='center',va='center',transform=axes[2].transAxes,color='gray')
        axes[2].set_title("PP_DINO\n(Bị bỏ qua)"); axes[2].axis('off')

    # Vẽ kết quả nối của PP1 (axes[3])
    visualize_matches_lines(img_design, img_real, matches_focused, "PP1: ElementMatcher (Để so sánh)", axes[3])

    fig.suptitle(f"So sánh Matching cho '{DESIGN_FILENAME}' và '{REAL_FILENAME}' (DINOv2 vs ElementMatcher)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    print("\n" + "="*20 + " DEMO HOÀN TẤT " + "="*20)

if __name__ == "__main__":
    try: main()
    except Exception as main_err:
        print(f"\nLỗi không xác định: {main_err}"); traceback.print_exc()