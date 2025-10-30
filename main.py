#!/usr/bin/env python3
"""
X-ray Analysis Main Script
Processes chest X-ray images and generates AI analysis reports
"""

import os
import sys
from datetime import datetime
from model_loader import ModelLoader
from image_processor import ImageProcessor
from text_generator import TextGenerator
from pdf_generator import PDFGenerator

def print_separator(title=""):
    """Print a decorative separator"""
    print("=" * 60)
    if title:
        print(f" {title} ".center(60))
        print("=" * 60)

def display_results(top3_diseases, explanation):
    """Display results in a formatted way"""
    print_separator("TOP 3 PREDICTED DISEASES")
    
    for i, (disease, probability) in enumerate(top3_diseases, 1):
        print(f"{i}. {disease}")
        print(f"   Confidence: {probability:.3f} ({probability*100:.1f}%)")
        print()
    
    print_separator("AI MEDICAL SUMMARY")
    print(explanation)
    print()
 
def main():
    # Configuration - Set your image path here
    IMAGE_PATH = r"E:/kakunje daily report/images/00006585_009.png"
    
    # Validate image path
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at {IMAGE_PATH}")
        return
    
    print_separator("CHEST X-RAY AI ANALYSIS")
    print(f"Processing: {os.path.basename(IMAGE_PATH)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Load models
        print("Step 1/5: Loading AI models...")
        model_loader = ModelLoader()
        models = model_loader.load_models().get_models()
        print()
        
        # Step 2: Process image
        print("Step 2/5: Processing X-ray image...")
        image_processor = ImageProcessor(models)
        image_results = image_processor.process_xray(IMAGE_PATH)
        print()
        
        # Step 3: Generate AI explanation
        print("Step 3/5: Generating AI medical summary...")
        text_generator = TextGenerator(models)
        explanation = text_generator.generate_explanation(image_results['top_disease'])
        print()
        
        # Step 4: Generate PDF report
        print("Step 4/5: Creating PDF report...")
        pdf_generator = PDFGenerator()
        top3_dict = dict(image_results['top3_diseases'])
        pdf_path = pdf_generator.generate_pdf(
            image_results['segmentation_image'], 
            top3_dict, 
            explanation
        )
        print()
        
        # Step 5: Display results
        print("Step 5/5: Analysis complete!")
        print()
        
        display_results(image_results['top3_diseases'], explanation)
        
        print_separator("REPORT SUMMARY")
        print(f"✓ Analysis completed successfully")
        print(f"✓ PDF report saved to: {pdf_path}")
        print(f"✓ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except FileNotFoundError as e:
        print(f"Error: Required model files not found - {e}")
        print("Please ensure all model files are in the correct directories:")
        print("- model8/unet_full_model.h5")
        print("- model8/densenet_xray_finetune_model3.h5")
        print("- mlb8/mlb_classes.pkl")
        print("- xray_gpt2_finetuned_model/")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check your image file and model paths.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    input("\nPress Enter to exit...")