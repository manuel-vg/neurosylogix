import utils

def tensor_to_list(tensor): 
    values = tensor.numpy()
    result = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        elif hasattr(value, "item"):
            value = value.item()
        result.append(value)

    return result

def set_of_premises(string_seq):
    return set([p.strip() for p in string_seq.split(",")])

def eval_model(model, tokenizer, test_dataset, output_file):
    # Lists to store predictions and their corresponding labels, ds_ids, hypotheses, types, and lenghts
    all_predictions = []
    all_labels = []
    all_ds_ids = []
    all_hypotheses = []
    all_types = []
    all_lengths = []

    # Evaluation
    for batch in test_dataset:
        labels = batch["labels"]
        ds_ids = batch["ds_ids"]
        hypotheses = batch["hypotheses"]
        types = batch["types"]
        lengths = batch["lengths"]

        tokenized = model.generate(input_ids=batch["input_ids"], max_new_tokens=256)
        predictions = tokenizer.batch_decode(tokenized, skip_special_tokens=True)

        # Append current batch
        all_predictions.extend(predictions)
        all_labels.extend(tensor_to_list(labels))
        all_ds_ids.extend(tensor_to_list(ds_ids))
        all_hypotheses.extend(tensor_to_list(hypotheses))
        all_types.extend(tensor_to_list(types))
        all_lengths.extend(tensor_to_list(lengths))

    # Compute global accuracy
    correct = sum(set_of_premises(p) == set_of_premises(l) for p, l in zip(all_predictions, all_labels))
    print(f"Accuracy: {correct / len(all_labels)}")

    # Save predictions        
    results = [
        {'prediction':pred, 'label':label, "ds_id":ds_id, "h":h, "type":type, "length":length} 
        for pred, label, ds_id, h, type, length in zip(all_predictions, all_labels, all_ds_ids, all_hypotheses, all_types, all_lengths)
    ]

    utils.save_json(json_path=output_file, data=results)
