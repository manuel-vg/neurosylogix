import argparse

from model import load_base_model
from dataset import Dataset
from train import train_model
from eval import eval_model

from pathlib import PurePath

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train","eval"], default="train")
parser.add_argument("--task", choices=["ps","pbc"], default="ps")
parser.add_argument("--experiment", choices=["ove","com","rec"], default="ove")
parser.add_argument("--run", type=int, default=1)
args = parser.parse_args()

# Configuration
NAME = f"T5_{args.task}_{args.experiment}" 
RUN = f"{NAME}_{args.run}"
BASE_MODEL = "google/flan-t5-base"
BATCH_SIZE_TRAIN = 20
BATCH_SIZE_EVAL = 256
EPOCHS = 1
LR = 1e-4
WEIGHT_DECAY = 0.004

# Dataset paths
TRAIN_JSON = f"data/{NAME}/train_{NAME}.json"
VALID_JSON = f"data/{NAME}/val_{NAME}.json"
TEST_JSON = [
    f"data/{NAME}/tds_{i}_{j}_{k}_{NAME}.json"
    for i in ["S", "M", "L"]
    for j in range(1, 11)
    for k in range(1, 4)
]

if args.mode == "train":
	# Load model and tokenizer
	model, tokenizer = load_base_model(model_name=BASE_MODEL)

	# Datasets
	train_tokenized = Dataset(json_path=TRAIN_JSON, tokenizer=tokenizer)
	valid_tokenized = Dataset(json_path=VALID_JSON, tokenizer=tokenizer)
	train_dataset = train_tokenized.train().shuffle(buffer_size=len(train_tokenized)).batch(BATCH_SIZE_TRAIN)
	valid_dataset = valid_tokenized.train().batch(BATCH_SIZE_TRAIN) 
    
	# Train model
	train_model(model, train_dataset, valid_dataset, EPOCHS, LR, WEIGHT_DECAY, RUN)

elif args.mode == "eval":
	# Load model (replacing fine-tuned weights) and tokenizer
	model, tokenizer = load_base_model(model_name=BASE_MODEL, weights=f"{RUN}.h5")

	# Datasets
	for test in TEST_JSON:				
		test_tokenized = Dataset(json_path=test, tokenizer=tokenizer)
		test_dataset = test_tokenized.evaluate().batch(BATCH_SIZE_EVAL)

		# Evaluation
		output_file = f"evaluations/{NAME}/run_{args.run}/pred_{PurePath(test).name}"
		eval_model(model, tokenizer, test_dataset, output_file)
