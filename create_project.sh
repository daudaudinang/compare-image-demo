#!/bin/bash
echo "Creating UI Comparator project structure..."

# Create main directories
mkdir -p ui_comparator/ui_comparator
mkdir -p ui_comparator/examples/sample_images

# Create empty files
touch ui_comparator/ui_comparator/__init__.py
touch ui_comparator/ui_comparator/preprocessing.py
touch ui_comparator/ui_comparator/segmentation.py
touch ui_comparator/ui_comparator/matching.py
touch ui_comparator/ui_comparator/analysis.py
touch ui_comparator/ui_comparator/visualization.py
touch ui_comparator/ui_comparator/utils.py
touch ui_comparator/examples/demo.py
touch ui_comparator/README.md

# Create environment.yml
cat > ui_comparator/environment.yml << EOL
name: ui-comparator
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pip
  - numpy
  - opencv
  - scikit-image
  - matplotlib
  - pytorch
  - pillow
  - scipy
  - pip:
    - paddlepaddle
    - paddleocr
    - transformers
EOL

echo "Project structure created successfully!"
chmod +x create_project.sh
