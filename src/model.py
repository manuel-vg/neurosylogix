from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer

def load_base_model(model_name, weights=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    if weights:
        model.load_weights(weights)

    return model, tokenizer
