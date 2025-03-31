# ui_comparator/examples/demo3.py (Sử dụng thư viện OpenAI để gọi API)

import os
import sys
import traceback
import json # Thêm thư viện json
import numpy as np # <<< THÊM DÒNG NÀY ĐỂ SỬA LỖI >>>

import matplotlib
matplotlib.use('TkAgg')

# --- Import cần thiết cho OpenAI và xử lý ảnh ---
try:
    from openai import OpenAI
    import base64
    from PIL import Image
    import io
    import cv2
except ImportError:
    print("Lỗi: Vui lòng cài đặt openai, Pillow, opencv-python")
    print("Chạy: pip install --upgrade openai pillow opencv-python")
    sys.exit(1)
# ----------------------------------------------

# Thêm thư mục cha vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui_comparator.preprocessing import Preprocessor
from ui_comparator.segmentation2 import UISegmenter
from ui_comparator.matching2 import ElementMatcher
from ui_comparator.analysis import DifferenceAnalyzer
from ui_comparator.visualization2 import ResultVisualizer # Đã sửa ở lần trước

# --- Cấu hình Endpoint tương thích OpenAI ---
OPENAI_COMPATIBLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY_FOR_ENDPOINT = "AIzaSyBT8qL7FtwTs0zcnYaV52gvk4Qg42mB46A" # <<< THAY KEY ĐÚNG
MODEL_NAME_FOR_ENDPOINT = "gemini-2.0-flash" # Đảm bảo model này hỗ trợ vision và JSON output tốt

# Khởi tạo OpenAI client
client = None
openai_configured = False
if API_KEY_FOR_ENDPOINT and API_KEY_FOR_ENDPOINT != "YOUR_API_KEY_FOR_THE_ENDPOINT" \
   and OPENAI_COMPATIBLE_BASE_URL:
    try:
        client = OpenAI(
            api_key=API_KEY_FOR_ENDPOINT,
            base_url=OPENAI_COMPATIBLE_BASE_URL,
        )
        print(f"Đã khởi tạo OpenAI client tới endpoint: {OPENAI_COMPATIBLE_BASE_URL}")
        openai_configured = True
    except Exception as e:
        print(f"Lỗi khởi tạo OpenAI client: {e}")
        print("Phần phân tích chi tiết bằng VLM sẽ bị bỏ qua.")
else:
    print("\nCẢNH BÁO: Chưa cấu hình Endpoint/API Key. Phân tích VLM sẽ bị bỏ qua.\n")


# --- Hàm encode ảnh sang base64 ---
def encode_image_to_base64(image_np):
    """Encodes a NumPy image array (BGR) to base64 data URI."""
    if image_np is None or image_np.size == 0: return None
    try:
        img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_image}"
    except Exception as e:
        print(f"   Lỗi encode ảnh: {e}")
        return None

# --- Hàm gọi VLM qua endpoint tương thích (ĐÃ CẬP NHẬT) ---
def analyze_difference_with_openai_compat_vlm(roi1_np, roi2_np):
    """Calls a VLM via OpenAI-compatible endpoint, requesting JSON output."""
    if not openai_configured or client is None:
        return "Client chưa được cấu hình." # Trả về string nếu lỗi cấu hình
    if roi1_np is None or roi2_np is None:
        return "Lỗi: Ảnh ROI đầu vào không hợp lệ." # Trả về string nếu lỗi ảnh

    try:
        base64_img1_uri = encode_image_to_base64(roi1_np)
        base64_img2_uri = encode_image_to_base64(roi2_np)

        if not base64_img1_uri or not base64_img2_uri:
            return "Lỗi: Không thể encode ảnh ROI." # Trả về string nếu lỗi encode

        # <<< PROMPT ĐÃ CẬP NHẬT (Tiếng Anh, Role-playing, JSON Output, Example) >>>
        prompt_text = """
You are a meticulous Visual QA Engineer. Your task is to compare two UI element images: Image 1 (Design Specification) and Image 2 (Actual Implementation).
Identify all significant visual and textual differences in Image 2 compared to Image 1. Focus on discrepancies in:
- Text Content: Typos, different wording, extra/missing text.
- Font & Size: Different font family, style (bold, italic), or size.
- Color: Differences in text color, background color, icon color, border color.
- Size & Dimensions: Noticeable differences in width or height.
- Position & Layout: Misalignments, incorrect spacing, different internal arrangement.
- Icons/Images: Different icons used, variations in image appearance.
- Missing/Extra Elements: Components present in one image but not the other (within the provided element scope).

Format your findings STRICTLY as a JSON object containing a single key "differences", which is a list of objects. Each difference object must have the following keys:
- "category": (string) One of ["Text Content", "Font & Size", "Color", "Size & Dimensions", "Position & Layout", "Icons/Images", "Missing/Extra Elements", "Other"]
- "description": (string) A concise description of the difference found in Image 2 compared to Image 1.
- "severity": (string) Estimated severity: "Low", "Medium", or "High".

If there are NO significant differences, return a JSON with an empty "differences" list: {"differences": []}.

Example:
Input: [Image 1 shows a blue 'Submit' button, Image 2 shows an orange 'Submit' button]
Output JSON:
{
  "differences": [
    {
      "category": "Color",
      "description": "Button background color in Image 2 (orange) differs from Image 1 (blue).",
      "severity": "Medium"
    }
  ]
}

Now, analyze the provided Image 1 and Image 2.
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": base64_img1_uri}},
                    {"type": "image_url", "image_url": {"url": base64_img2_uri}}
                ]
            }
        ]

        print(f"      Đang gọi endpoint ({MODEL_NAME_FOR_ENDPOINT}) yêu cầu JSON...")
        response = client.chat.completions.create(
            model=MODEL_NAME_FOR_ENDPOINT,
            messages=messages,
            max_tokens=500, # Tăng token limit một chút cho JSON
            temperature=0.0 # <<< SET TEMPERATURE = 0 >>>
            # response_format={"type": "json_object"} # Bỏ comment nếu endpoint hỗ trợ erz erzwingen JSON
        )

        raw_response_content = response.choices[0].message.content if response.choices else None

        if not raw_response_content:
            return "Lỗi: Phản hồi VLM rỗng." # Trả về string nếu phản hồi rỗng

        # Cố gắng parse JSON
        try:
            # Loại bỏ ```json và ``` nếu có (một số model hay thêm vào)
            cleaned_content = raw_response_content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()

            # Parse JSON
            parsed_json = json.loads(cleaned_content)

            # Kiểm tra cấu trúc cơ bản (có key 'differences' là list không)
            if isinstance(parsed_json, dict) and isinstance(parsed_json.get('differences'), list):
                return parsed_json # Trả về đối tượng JSON đã parse thành công
            else:
                print("      Warning: VLM response JSON structure is incorrect. Returning raw text.")
                return f"Lỗi JSON Structure: {raw_response_content}" # Trả về string lỗi cấu trúc

        except json.JSONDecodeError as json_err:
            print(f"      Warning: Failed to parse VLM response as JSON: {json_err}")
            return f"Lỗi JSON Decode: {raw_response_content}" # Trả về string lỗi parse

    except Exception as e:
        print(f"      Lỗi khi gọi endpoint hoặc xử lý VLM: {e}")
        # print(traceback.format_exc())
        return f"Lỗi VLM Exception: {type(e).__name__}" # Trả về string lỗi chung
# --- Kết thúc hàm gọi VLM ---


def main():
    """Chạy demo so sánh UI."""
    design_filename = 'design1.png'
    real_filename = 'real1.jpg'
    design_path = os.path.join('sample_images', design_filename)
    real_path = os.path.join('sample_images', real_filename)

    if not os.path.exists(design_path): print(f"Lỗi: Ko tìm thấy Design: {design_path}"); return
    if not os.path.exists(real_path): print(f"Lỗi: Ko tìm thấy Real: {real_path}"); return

    print(f"=== BẮT ĐẦU SO SÁNH UI ({design_filename} vs {real_filename}) ===")

    # 1. Tiền xử lý ảnh
    print("\nBước 1: Tiền xử lý ảnh...")
    preprocessor = Preprocessor()
    try:
        processed_design, processed_real = preprocessor.process(design_path, real_path)
        print(f"   Ảnh sau tiền xử lý: Design{processed_design.shape}, Real{processed_real.shape}")
    except Exception as e: print(f"   Lỗi tiền xử lý: {e}. Dừng."); return

    # 2. Phân đoạn UI
    print("\nBước 2: Phân đoạn UI...")
    segmenter = UISegmenter()
    try:
        elements_design = segmenter.segment(processed_design)
        elements_real = segmenter.segment(processed_real)
        def assign_ids_and_rois(elements, img, prefix):
             img_h, img_w = img.shape[:2]
             for i, elem in enumerate(elements):
                 elem['id'] = f"{prefix}{i}"
                 try:
                     bbox = elem.get('bbox')
                     if not isinstance(bbox, (tuple, list)) or len(bbox) != 4: elem['roi'] = None; continue
                     x, y, w, h = map(int, map(float, bbox))
                     y_start, y_end = max(0, y), min(img_h, y + h)
                     x_start, x_end = max(0, x), min(img_w, x + w)
                     elem['roi'] = img[y_start:y_end, x_start:x_end].copy() if y_end > y_start and x_end > x_start else None
                 except (ValueError, TypeError) as assign_err:
                     print(f"   Warning: Lỗi gán ROI cho {prefix}{i}: {assign_err}, bbox: {elem.get('bbox')}")
                     elem['roi'] = None
             return elements
        elements_design = assign_ids_and_rois(elements_design, processed_design, 'D')
        elements_real = assign_ids_and_rois(elements_real, processed_real, 'R')
        print(f"   Phân đoạn xong: Design({len(elements_design)}), Real({len(elements_real)}).")
    except Exception as e: print(f"   Lỗi phân đoạn: {e}. Dừng."); print(traceback.format_exc()); return

    # 3. So khớp thành phần UI
    print("\nBước 3: So khớp thành phần UI (DINOv2)...")
    matcher_weights = {'dino': 0.6, 'pos': 0.3, 'size': 0.1}
    matcher_combined_threshold = 0.55
    matcher_min_pos_threshold = 0.4
    matcher = ElementMatcher()
    try:
        matches = matcher.match_elements(
            elements_design, elements_real, processed_design.shape, processed_real.shape,
            weights=matcher_weights,
            combined_threshold=matcher_combined_threshold,
            min_pos_threshold=matcher_min_pos_threshold
        )
        print(f"   Tìm thấy {len(matches)} cặp khớp.")
    except Exception as e: print(f"   Lỗi matching: {e}. Dừng."); print(traceback.format_exc()); return

    # ***** PHÂN TÍCH CHI TIẾT BẰNG VLM (ĐÃ CẬP NHẬT) *****
    print("\nBước 3.5: Phân tích chi tiết các cặp 'gần giống' bằng VLM...")
    detailed_vlm_analysis = []
    if openai_configured:
        num_analyzed_vlm = 0
        MAX_VLM_PAIRS = 15
        vlm_similarity_threshold_low = matcher_combined_threshold
        vlm_similarity_threshold_high = 0.92

        processed_count = 0
        for match in matches:
            similarity = match.get('similarity', 0.0)
            if vlm_similarity_threshold_low < similarity <= vlm_similarity_threshold_high:
                processed_count += 1
                if num_analyzed_vlm >= MAX_VLM_PAIRS:
                    print(f"      Đã đạt giới hạn {MAX_VLM_PAIRS} cặp gọi VLM analysis.")
                    break

                element1 = match.get('element1')
                element2 = match.get('element2')
                if element1 and element2:
                    roi1 = element1.get('roi')
                    roi2 = element2.get('roi')
                    if roi1 is not None and roi1.size > 0 and roi2 is not None and roi2.size > 0:
                        print(f"      ({processed_count}) Phân tích VLM cho cặp: {element1.get('id', '?')} vs {element2.get('id', '?')} (Sim: {similarity:.3f})")

                        # Gọi hàm VLM (đã cập nhật để trả về JSON hoặc string lỗi)
                        vlm_result = analyze_difference_with_openai_compat_vlm(roi1, roi2)

                        # In ra kết quả nhận được (JSON hoặc lỗi)
                        if isinstance(vlm_result, dict):
                             print(f"        => VLM JSON received (differences: {len(vlm_result.get('differences', []))})")
                        else:
                             print(f"        => VLM Error/Raw: {vlm_result[:150]}...") # In phần đầu nếu là lỗi string

                        detailed_vlm_analysis.append({
                            'element1_id': element1.get('id'),
                            'element2_id': element2.get('id'),
                            'similarity': similarity,
                            'vlm_analysis': vlm_result # Lưu kết quả JSON hoặc string lỗi
                        })
                        num_analyzed_vlm += 1
                    else:
                        print(f"      Cảnh báo: Bỏ qua cặp {element1.get('id', '?')} / {element2.get('id', '?')} do thiếu ROI hợp lệ.")
        print(f"   Đã phân tích {num_analyzed_vlm} cặp bằng VLM.")
    else:
        print("   Bỏ qua phân tích VLM do Endpoint/API Key chưa được cấu hình.")
    # ************************************************************

    # 4. Phân tích sự khác biệt cơ bản
    print("\nBước 4: Phân tích sự khác biệt cơ bản...")
    analyzer = DifferenceAnalyzer()
    analysis_result = {} # Khởi tạo trước
    ssim_score = 0.0
    # Khởi tạo diff_heatmap với giá trị mặc định phòng trường hợp lỗi xảy ra trước khi nó được gán giá trị từ compare_structure
    # Sử dụng np đã import ở đầu file
    diff_heatmap = np.zeros_like(processed_design) if 'processed_design' in locals() and processed_design is not None else None

    try:
        # Đảm bảo processed_design và processed_real tồn tại trước khi gọi compare_structure
        if 'processed_design' in locals() and processed_design is not None and \
           'processed_real' in locals() and processed_real is not None:
            ssim_score, diff_map, diff_heatmap = analyzer.compare_structure(processed_design, processed_real)
            print(f"   Điểm tương đồng cấu trúc (SSIM): {ssim_score:.3f}")
        else:
            print("   Cảnh báo: Không thể tính SSIM do thiếu ảnh đã xử lý.")
            ssim_score = 0.0 # Gán giá trị mặc định

        # Phân tích element differences luôn được thực hiện nếu có matches, elements1, elements2
        if 'matches' in locals() and 'elements_design' in locals() and 'elements_real' in locals():
             analysis_result = analyzer.analyze_element_differences(matches, elements_design, elements_real)
             # Gắn kết quả VLM vào
             analysis_result['vlm_details'] = detailed_vlm_analysis # Đổi tên key cho nhất quán với visualization
             print(f"   Phân tích xong: {len(analysis_result.get('matched_elements',[]))} matched, {len(analysis_result.get('missing_elements',[]))} missing, {len(analysis_result.get('extra_elements',[]))} extra.")
        else:
             print("   Cảnh báo: Không thể phân tích element differences do thiếu dữ liệu đầu vào.")
             # Đảm bảo analysis_result có cấu trúc cơ bản
             analysis_result.setdefault('matched_elements', [])
             analysis_result.setdefault('missing_elements', [])
             analysis_result.setdefault('extra_elements', [])
             analysis_result['vlm_details'] = detailed_vlm_analysis


    except Exception as e:
        print(f"   Lỗi phân tích: {e}."); print(traceback.format_exc())
        # Đảm bảo analysis_result có key 'vlm_details' dù lỗi
        analysis_result.setdefault('matched_elements', [])
        analysis_result.setdefault('missing_elements', [])
        analysis_result.setdefault('extra_elements', [])
        analysis_result['vlm_details'] = detailed_vlm_analysis
        # Đảm bảo diff_heatmap tồn tại nếu lỗi xảy ra sau khi nó được khởi tạo
        if diff_heatmap is None and 'processed_design' in locals() and processed_design is not None:
             diff_heatmap = np.zeros_like(processed_design)


    # 5. Hiển thị kết quả (bao gồm tạo báo cáo HTML)
    print("\nBước 5: Hiển thị kết quả và tạo báo cáo HTML...")
    visualizer = ResultVisualizer()
    try:
        # Kiểm tra diff_heatmap trước khi truyền vào
        if diff_heatmap is None:
             print("   Cảnh báo: Không có heatmap để hiển thị.")
             # Có thể tạo heatmap rỗng nếu cần thiết cho hàm display_results không bị lỗi
             if 'processed_design' in locals() and processed_design is not None:
                 diff_heatmap = np.zeros_like(processed_design)
             else: # Trường hợp xấu nhất, không có ảnh gốc để tạo heatmap rỗng
                  diff_heatmap = np.zeros((100,100,3), dtype=np.uint8) # Tạo ảnh đen nhỏ

        visualizer.display_results(
            processed_design, processed_real, elements_design, elements_real,
            matches, analysis_result, ssim_score, diff_heatmap,
            match_threshold=matcher_combined_threshold
        )
    except Exception as e: print(f"   Lỗi hiển thị kết quả / tạo báo cáo: {e}"); print(traceback.format_exc())

    print("\n=== KẾT THÚC SO SÁNH UI ===")

if __name__ == "__main__":
    main()
