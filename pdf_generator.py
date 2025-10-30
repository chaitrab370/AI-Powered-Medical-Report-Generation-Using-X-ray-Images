import os
import tempfile
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

class PDFGenerator:
    def __init__(self, output_dir="E:\\232008chaitraAIxrayproject(flask)\\pdf_report"):
        self.output_dir = output_dir
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_pdf(self, seg_img, top3_dict, explanation, custom_filename=None):
        """Generate PDF report with segmentation image and predictions"""
        print("Generating PDF report...")
        
        # Create filename with timestamp if not provided
        if custom_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Xray_Report_{timestamp}.pdf"
        else:
            filename = custom_filename
            
        pdf_path = os.path.join(self.output_dir, filename)
        
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Chest X-ray AI Analysis Report")
        
        # Date and time
        c.setFont("Helvetica", 10)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.drawString(50, height - 70, f"Generated on: {current_time}")
        
        # Top 3 diseases section
        c.setFont("Helvetica-Bold", 12)
        y = height - 110
        c.drawString(50, y, "Top 3 Predicted Diseases and Probabilities:")
        
        c.setFont("Helvetica", 11)
        y -= 25
        for i, (disease, prob) in enumerate(top3_dict.items(), 1):
            c.drawString(70, y, f"{i}. {disease}: {prob:.3f} ({prob*100:.1f}%)")
            y -= 20
        
        # AI explanation section
        if explanation:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y - 20, "AI-Generated Medical Summary:")
            y -= 45
            c.setFont("Helvetica", 10)
            
            # Word wrap the explanation
            wrapped_text = textwrap.wrap(explanation, width=95)
            for line in wrapped_text:
                if y < 100:  # Start new page if needed
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = height - 50
                c.drawString(70, y, line)
                y -= 15
        
        # Add segmentation image on new page
        if seg_img:
            c.showPage()
            
            # Save segmentation image temporarily
            seg_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            seg_img.save(seg_path)
            
            # Add image to PDF
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height - 50, "Lung Segmentation Analysis")
            
            # Add image (adjust size to fit page)
            try:
                c.drawImage(seg_path, 50, height - 450, width=400, height=300, preserveAspectRatio=True)
            except Exception as e:
                print(f"Warning: Could not add image to PDF: {e}")
            
            # Add description
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 480, "Above: AI-generated lung segmentation mask highlighting the lung regions")
            
            # Clean up temporary file
            try:
                os.remove(seg_path)
            except:
                pass
        
        # Save PDF
        c.save()
        print(f"✓ PDF report saved to: {pdf_path}")
        
        return pdf_path