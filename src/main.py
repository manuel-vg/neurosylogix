from config import *

import argparse

from model import load_base_model
from dataset import Dataset
from train import train_model
from eval import eval_model

from pathlib import PurePath

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train","eval"], default="train")
args = parser.parse_args()

if args.mode == "train":
	# Load model and tokenizer
	model, tokenizer = load_base_model(model_name=BASE_MODEL)

	# Datasets
	train_tokenized = Dataset(json_path=TRAIN_JSON, tokenizer=tokenizer)
	valid_tokenized = Dataset(json_path=VALID_JSON, tokenizer=tokenizer)
	train_dataset = train_tokenized.train().shuffle(buffer_size=len(train_tokenized)).batch(BATCH_SIZE_TRAIN)
	valid_dataset = valid_tokenized.train().batch(BATCH_SIZE_TRAIN) 
    
	# Train model
	train_model(model, train_dataset, valid_dataset, EPOCHS, LR, WEIGHT_DECAY, NAME)

elif args.mode == "eval":
	# Load model (replacing fine-tuned weights) and tokenizer
	model, tokenizer = load_base_model(model_name=BASE_MODEL, weights=f"{NAME}.h5")

	# Datasets
	for test in TEST_JSON:				
		test_tokenized = Dataset(json_path=test, tokenizer=tokenizer)
		test_dataset = test_tokenized.evaluate().batch(BATCH_SIZE_EVAL)

		# Evaluation
		output_file = f"{EVAL_FOLDER}/pred_{PurePath(test).name}"
		eval_model(model, tokenizer, test_dataset, output_file)