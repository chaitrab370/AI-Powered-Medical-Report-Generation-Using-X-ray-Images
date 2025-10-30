import numpy as np
from PIL import Image
import base64
from io import BytesIO

class ImageProcessor:
    def __init__(self, models_dict):
        self.unet_model = models_dict['unet_model']
        self.densenet_model = models_dict['densenet_model']
        self.class_labels = models_dict['class_labels']
        self.IMG_SIZE = models_dict['IMG_SIZE']
        self.THRESHOLD = models_dict['THRESHOLD']
    
    def predict_mask(self, image):
        """Generate lung segmentation mask"""
        image = image.convert("L").resize((self.IMG_SIZE, self.IMG_SIZE))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))
        pred_mask = self.unet_model.predict(img_array)[0, :, :, 0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        return Image.fromarray(pred_mask)
    
    def predict_diseases(self, image):
        """Predict diseases from X-ray image"""
        img_resized = image.resize((self.IMG_SIZE, self.IMG_SIZE))
        img_array = np.array(img_resized) / 255.0
        
        # Handle RGBA images
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        
        img_array = np.expand_dims(img_array, axis=0)
        preds = self.densenet_model.predict(img_array)[0]
        
        # Get top 3 predictions
        top3_idx = np.argsort(preds)[-3:]
        top3_sorted = sorted(
            [(self.class_labels[i], round(float(preds[i]), 3)) for i in top3_idx],
            key=lambda x: x[1],
            reverse=True
        )
        
        return top3_sorted
    
    def image_to_base64(self, img: Image.Image):
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def process_xray(self, image_path):
        """Complete X-ray processing pipeline"""
        print(f"Processing X-ray image: {image_path}")
        
        # Load and convert image
        image = Image.open(image_path).convert("RGB")
        print("✓ Image loaded and converted")
        
        # Predict diseases
        top3_diseases = self.predict_diseases(image)
        print("✓ Disease prediction completed")
        
        # Generate segmentation mask
        seg_img = self.predict_mask(image)
        print("✓ Lung segmentation completed")
        
        return {
            'top3_diseases': top3_diseases,
            'segmentation_image': seg_img,
            'top_disease': top3_diseases[0][0]
        }