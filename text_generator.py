import torch

class TextGenerator:
    def __init__(self, models_dict):
        self.tokenizer = models_dict['tokenizer']
        self.gpt2_model = models_dict['gpt2_model']
        self.device = models_dict['device']
    
    def generate_explanation(self, disease):
        """Generate AI explanation for the detected disease"""
        print(f"Generating explanation for: {disease}")
        
        prompt = f"<disease> {disease} <report>"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.gpt2_model.generate(
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
        
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        explanation = output_text.split("<report>")[-1].strip()
        
        print("✓ AI explanation generated")
        return explanation