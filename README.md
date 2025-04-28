# Triangle Detection using YOLOv8

![Triangle Detection Banner](https://img.shields.io/badge/Computer%20Vision-Triangle%20Detection-blue) ![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-darkgreen) ![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

This repository contains the implementation of a triangle detection system using YOLOv8, as part of the BigVision assessment. The project focuses on detecting triangles in images of varying sizes, colors, and backgrounds.

## 🌟 [Live Demo on Hugging Face Spaces - Click Here](https://huggingface.co/spaces/harxsan/Triangle-Detector)

Try the model in action without any setup! Upload any image and see the triangle detection results in real-time.

## 📝 Project Overview

This project implements a custom object detection system specifically trained to detect triangles in images. The approach includes:

1. **Synthetic Dataset Creation**: Generation of a diverse dataset with triangles of varying shapes, sizes, and colors on different backgrounds
2. **Model Training**: Training a YOLOv8 model optimized for triangle detection 
3. **Evaluation**: Comprehensive evaluation of model performance with precision, recall, and mAP metrics
4. **Deployment**: User-friendly Gradio interface for real-time triangle detection

## 🔍 Project Structure

The project consists of two main notebooks:

1. **[Triangle Detection Training](BigVision_Task_Triangle_Detection.py)** - Creates the dataset, trains the model, and saves it to Google Drive
2. **[Triangle Detection UI](Triangle_Detection_UI.ipynb)** - Loads the trained model from Google Drive and provides a Gradio interface for testing

## 🧠 Technical Implementation Details

### Dataset Generation

The synthetic dataset includes:

- **Total images**: 1500 (split into train/validation/test)
- **Background variety**: Solid colors, gradients, noise patterns, and textures
- **Triangle types**: Regular, thin, obtuse, and right triangles
- **Negative examples**: 20% of images contain no triangles for robust training
- **Image size**: 640×640 pixels

### Model Architecture and Training

- **Model**: YOLOv8n (nano) for efficiency and speed
- **Training parameters**:
  - Epochs: 100 with early stopping patience of 20
  - Batch size: 16
  - Extensive data augmentation: rotations, flips, color jitter, etc.
- **Optimization**: AdamW optimizer with automatic learning rate

### Performance Metrics

The trained model achieves:
- mAP@50: ~0.95
- Precision: ~0.93
- Recall: ~0.90
- Fast inference: ~5-10ms per image on GPU

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Ultralytics (YOLOv8)
- OpenCV
- Albumentations
- Google Colab (recommended for training)

### Using the Training Notebook

1. Open `BigVision_Task_Triangle_Detection.py` in Google Colab
2. Mount your Google Drive: `drive.mount('/content/drive')`
3. Set the `SAVE_DIR` path in the CFG class to your desired location
4. Run all cells to generate the dataset and train the model
5. The best model will be saved to your Google Drive

### Using the Inference UI Notebook

1. Open `Triangle_Detection_UI.ipynb` in Google Colab
2. Mount your Google Drive where the model is saved
3. Update the path to the saved model
4. Run all cells to start the Gradio interface
5. Upload images to test triangle detection

### Alternative: Use the Hugging Face Space

Visit [Triangle-Detector](https://huggingface.co/spaces/harxsan/Triangle-Detector) to use the pre-trained model directly in your browser.

## 📊 Results Visualization

The training process generates several visualizations:
- Training curves (loss, precision, recall, mAP)
- Dataset distribution statistics
- Example detections on test images

## 📱 Deployment 

The model is deployed using Gradio, providing:
- Simple user interface for image upload
- Real-time triangle detection
- Visualization of detected triangles with bounding boxes
- Confidence scores for each detection

## 🛠️ Future Improvements

- Collection of real-world triangle images for fine-tuning
- Testing with larger YOLOv8 models (s, m, l) for higher accuracy
- Integration with video processing for real-time detection
- Additional geometric shape classes (squares, circles, etc.)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Contact

Feel free to connect with me for any questions or collaboration opportunities!

- GitHub: [HarxSan](https://github.com/HarxSan)
- Hugging Face: [harxsan](https://huggingface.co/harxsan)

---

*This project was created as part of the BigVision assessment for custom object detection*
