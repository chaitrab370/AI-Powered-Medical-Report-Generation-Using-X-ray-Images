from tensorflow.keras.models import load_model

# Load the complete model
unet_model8 = load_model('unet_full_model.h5')

# Check the model summary
unet_model8.summary()

#unet_model = load_model('/content/drive/MyDrive/unet_full_model.h5')