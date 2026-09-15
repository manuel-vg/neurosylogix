import tensorflow as tf
from utils import open_json

class Dataset:
    def __init__(self, json_path, tokenizer):
        self.data = open_json(json_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _inputs(self):
        return [f"knowledge base: {item['kb']} hypothesis: {item['h']}" for item in self.data]

    def _targets(self):
        return [item["target"] for item in self.data]

    def _preprocess(self, mode="train"):
        inputs = self.tokenizer(self._inputs(), padding=True, return_tensors="tf")
        if mode == "eval":
            return dict(inputs)
        labels = self.tokenizer(self._targets(), padding=True, return_tensors="tf")
        inputs["labels"] = labels["input_ids"]
        
        return dict(inputs)

    def train(self):       
        tokenized = self._preprocess() 

        return tf.data.Dataset.from_tensor_slices(tokenized)

    def evaluate(self):
        evaluation_data = {**self._preprocess(mode="eval"),
            "labels": self._targets(), 
            "ds_ids": [item["ds_id"] for item in self.data],
            "hypotheses": [item["h"] for item in self.data],
            "types": [item["type"] for item in self.data],
            "lengths": [item["length"] for item in self.data]
        }

        return tf.data.Dataset.from_tensor_slices(evaluation_data)
