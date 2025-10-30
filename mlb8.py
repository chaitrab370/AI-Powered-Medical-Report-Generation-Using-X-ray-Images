import joblib

mlb = joblib.load('mlb_classes.pkl')
class_labels = mlb.classes_

print(f"🔁 Loaded mlb with {len(class_labels)} classes")