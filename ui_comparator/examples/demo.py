"""Demo cho UI Comparator."""

import os
import sys

import matplotlib

matplotlib.use('TkAgg')  # Sử dụng backend TkAgg

# Thêm thư mục cha vào sys.path để import được module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui_comparator.preprocessing import Preprocessor
from ui_comparator.segmentation2 import UISegmenter
from ui_comparator.matching2 import ElementMatcher
from ui_comparator.analysis import DifferenceAnalyzer
from ui_comparator.visualization2 import ResultVisualizer

def main():
    """Chạy demo so sánh UI."""
    # Đường dẫn đến ảnh mẫu
    design_path = os.path.join('sample_images', 'design2.jpg')
    real_path = os.path.join('sample_images', 'real2.jpg')
    
    print("=== BẮT ĐẦU SO SÁNH UI ===")
    
    # 1. Tiền xử lý ảnh
    print("Bước 1: Tiền xử lý ảnh...")
    preprocessor = Preprocessor()
    processed_design, processed_real = preprocessor.process(design_path, real_path)
    
    # 2. Phân đoạn UI
    print("Bước 2: Phân đoạn UI...")
    segmenter = UISegmenter()
    elements_design = segmenter.segment(processed_design)
    elements_real = segmenter.segment(processed_real)
    
    # 3. So khớp thành phần UI
    print("Bước 3: So khớp thành phần UI...")
    matcher = ElementMatcher()
    matches = matcher.match_elements(elements_design, elements_real, 
                                    processed_design.shape, processed_real.shape)
    
    # 4. Phân tích sự khác biệt
    print("Bước 4: Phân tích sự khác biệt...")
    analyzer = DifferenceAnalyzer()
    ssim_score, diff_map, diff_heatmap = analyzer.compare_structure(processed_design, processed_real)
    analysis_result = analyzer.analyze_element_differences(matches, elements_design, elements_real)
    
    # 5. Hiển thị kết quả
    print("Bước 5: Hiển thị kết quả...")
    visualizer = ResultVisualizer()
    visualizer.display_results(
        processed_design, processed_real,
        elements_design, elements_real,
        matches, analysis_result,
        ssim_score, diff_heatmap
    )
    
    print("=== KẾT THÚC SO SÁNH UI ===")

if __name__ == "__main__":
    main()
