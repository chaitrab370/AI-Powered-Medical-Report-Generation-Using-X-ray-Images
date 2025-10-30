import os
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import tensorflow as tf
import joblib
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tensorflow.keras.models import load_model
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import textwrap
import torch
import base64
from io import BytesIO

app = Flask(__name__)

# --- Load models and tokenizers ---
def focal_loss(gamma=2., alpha=.25):
    def loss_fn(y_true, y_pred):
        epsilon = 1e-9
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss)
    return loss_fn

IMG_SIZE = 256
THRESHOLD = 0.2

unet_model = load_model('model8/unet_full_model.h5', compile=False)
densenet_model = load_model('model8/densenet_xray_finetune_model3.h5', compile=False)
mlb = joblib.load('mlb8/mlb_classes.pkl')
class_labels = mlb.classes_

# GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("xray_gpt2_finetuned_model")
gpt2_model = GPT2LMHeadModel.from_pretrained("xray_gpt2_finetuned_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model.to(device)

# --- Utility Functions ---
def predict_mask(image):
    image = image.convert("L").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    pred_mask = unet_model.predict(img_array)[0, :, :, 0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    return Image.fromarray(pred_mask)

def generate_explanation(disease):
    prompt = f"<disease> {disease} <report>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = gpt2_model.generate(
            input_ids,
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    explanation = output_text.split("<report>")[-1].strip()
    return explanation

def generate_pdf(seg_img, top3_dict, explanation):
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Chest X-ray Top 3 Disease Report")
    c.setFont("Helvetica", 12)
    y = height - 100
    c.drawString(50, y, "Top 3 Predicted Diseases and Probabilities:")
    y -= 20
    for disease, prob in top3_dict.items():
        c.drawString(70, y, f"{disease}: {prob:.3f}")
        y -= 20

    if explanation:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y - 10, "AI-Generated Medical Report:")
        y -= 30
        c.setFont("Helvetica", 11)
        for line in textwrap.wrap(explanation, width=95):
            if y < 100:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - 50
            c.drawString(70, y, line)
            y -= 20

    if seg_img:
        seg_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        seg_img.save(seg_path)
        c.showPage()
        c.drawImage(seg_path, 50, 300, width=400, preserveAspectRatio=True)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 270, "Segmented Lung Mask")
        c.save()
        os.remove(seg_path)
    else:
        c.save()

    return pdf_path

def image_to_base64(img: Image.Image):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def analyze():
    file = request.files['xrayImage']
    image = Image.open(file.stream).convert("RGB")
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)

    preds = densenet_model.predict(img_array)[0]
    top3_idx = np.argsort(preds)[-3:]
    top3_sorted = sorted(
        [(class_labels[i], round(float(preds[i]), 3)) for i in top3_idx],
        key=lambda x: x[1],
        reverse=True
    )
    top_disease = top3_sorted[0][0]

    explanation = generate_explanation(top_disease)
    seg_img = predict_mask(image)
    seg_base64 = image_to_base64(seg_img)

    pdf_path = generate_pdf(seg_img, dict(top3_sorted), explanation)

    # --- These lines print to terminal ---
    print("Top 3 Predicted Diseases:", top3_sorted)
    print("Explanation Generated by GPT-2:\n", explanation)
    print("PDF Report Path:", pdf_path)

    return jsonify({
        "top3": top3_sorted,
        "explanation": explanation,
        "segmentationImage": f"data:image/png;base64,{seg_base64}",
        "pdf_url": f"/download_pdf?path={pdf_path}"
    })


@app.route('/download_pdf')
def download_pdf():
    pdf_path = request.args.get('path')
    return send_file(pdf_path, as_attachment=True, download_name="Xray_Report.pdf")

if __name__ == '__main__':
    app.run(debug=True)


