from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_tokenizer = None
_model = None

def load_model():
    global _tokenizer, _model
    if _tokenizer is None:
        print("Loading local MBZUAI/LaMini-Flan-T5-783M model...")
        model_id = "MBZUAI/LaMini-Flan-T5-783M"
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        print("Local LaMini model loaded successfully!")
    return _tokenizer, _model

def get_concept_explanation(prompt: str) -> str:
    if not prompt or not prompt.strip():
        return "Please provide a concept or topic to explain."

    try:
        tokenizer, model = load_model()
        system_input = f"Explain this concept clearly and concisely: {prompt.strip()}"

        inputs = tokenizer(system_input, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    except Exception as e:
        return f"Error running local model: {str(e)}"