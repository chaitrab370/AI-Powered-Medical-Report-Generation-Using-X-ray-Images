import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

class ModelLoader:
    def __init__(self):
        self.IMG_SIZE = 256
        self.THRESHOLD = 0.2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.unet_model = None
        self.densenet_model = None
        self.mlb = None
        self.class_labels = None
        self.tokenizer = None
        self.gpt2_model = None
        
    def focal_loss(self, gamma=2., alpha=.25):
        def loss_fn(y_true, y_pred):
            epsilon = 1e-9
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            cross_entropy = -y_true * tf.math.log(y_pred)
            loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
            return tf.reduce_mean(loss)
        return loss_fn
    
    def load_models(self):
        """Load all required models"""
        print("Loading models...")
        
        # Load UNet model
        self.unet_model = load_model('model8/unet_full_model.h5', compile=False)
        print("✓ UNet model loaded")
        
        # Load DenseNet model
        self.densenet_model = load_model('model8/densenet_xray_finetune_model3.h5', compile=False)
        print("✓ DenseNet model loaded")
        
        # Load label encoder
        self.mlb = joblib.load('mlb8/mlb_classes.pkl')
        self.class_labels = self.mlb.classes_
        print("✓ Label encoder loaded")
        
        # Load GPT-2 model
        self.tokenizer = GPT2Tokenizer.from_pretrained("xray_gpt2_finetuned_model")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("xray_gpt2_finetuned_model")
        self.gpt2_model.to(self.device)
        print("✓ GPT-2 model loaded")
        
        print("All models loaded successfully!")
        return self
    
    def get_models(self):
        """Return all loaded models"""
        return {
            'unet_model': self.unet_model,
            'densenet_model': self.densenet_model,
            'mlb': self.mlb,
            'class_labels': self.class_labels,
            'tokenizer': self.tokenizer,
            'gpt2_model': self.gpt2_model,
            'device': self.device,
            'IMG_SIZE': self.IMG_SIZE,
            'THRESHOLD': self.THRESHOLD
        }