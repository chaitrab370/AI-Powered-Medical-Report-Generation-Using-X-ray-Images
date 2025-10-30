##Developed an AI-based medical report generation system using DenseNet121, U-Net, GPT-2, achieving 82.9% AUC. 
Automated report creation via Gradio and Flask, generating structured PDF outputs for radiology use.

AI-Powered Medical Report Generation using X-ray Images
## Overview
This project is implemented using Python as the primary programming language. It leverages powerful Deep Learning frameworks such as TensorFlow, Keras, and PyTorch for model training and inference. For Natural Language Processing (NLP), models like GPT-2 and BERT are utilized to generate structured and meaningful medical reports. The visual feature extraction is performed using DenseNet121 as the CNN backbone, while U-Net is employed for image segmentation to highlight specific regions of interest in X-ray images.

The system provides a user-friendly interface built with Gradio or Flask, enabling easy interaction with the AI models. Visualization of predictions and segmentations is achieved through Matplotlib and OpenCV. The model is trained and tested on the Indiana University Chest X-ray Dataset (Kaggle), which includes X-ray images and their corresponding medical findings.

## Project Workflow

# Image Upload
   User uploads an X-ray image (PNG or DICOM).
# Preprocessing & Segmentation (U-Net)
   The uploaded image is segmented to isolate the lungs or specific organ region.
# Disease Classification (DenseNet121)
   The system predicts the top probable diseases or abnormalities.
# Report Generation (GPT-2 / BERT)
   A language model generates a structured radiology report with findings and impressions.
# Report Output
   The generated report can be viewed, downloaded as PDF, or shared directly via the interface.

## Model Details
DenseNet121 was fine-tuned for multi-label classification on the Indiana X-ray dataset.
U-Net was trained for lung segmentation to highlight affected areas.
GPT-2 / BERT was fine-tuned for radiology-style report text generation.

