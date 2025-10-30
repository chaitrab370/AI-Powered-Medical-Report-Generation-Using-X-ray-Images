import tensorflow as tf
from tensorflow.keras.models import load_model


# Now pass it to load_model with the exact identifier
densenet_model8 = load_model('densenet_xray_finetune_model3.h5', compile=False)
densenet_model8.summary()

#model3 = load_model("/content/drive/MyDrive/densenet_xray_finetune_model3.h5", custom_objects={'loss_fn': focal_loss()})