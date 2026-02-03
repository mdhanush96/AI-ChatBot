import json
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import classification_report, f1_score

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

TRAIN_FILE = "data/train_final.json"
VAL_FILE = "data/val_ncbi.json"
TEST_FILE = "data/test_ncbi.json"

OUTPUT_DIR = "./biobert_ner_model"

LABEL_LIST = ["O", "B-DISEASE", "I-DISEASE"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5

# -------------------------------
# LOAD DATA
# -------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

train_data = load_json(TRAIN_FILE)
val_data = load_json(VAL_FILE)
test_data = load_json(TEST_FILE)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)

# -------------------------------
# TOKENIZER
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(LABEL2ID[label[word_idx]])
            else:
                label_ids.append(LABEL2ID[label[word_idx]])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

# -------------------------------
# MODEL
# -------------------------------
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# -------------------------------
# METRICS
# -------------------------------
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    return {
        "f1": f1_score(true_labels, true_predictions)
    }

# -------------------------------
# TRAINING ARGUMENTS
# -------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# -------------------------------
# TRAINER
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# -------------------------------
# TRAIN
# -------------------------------
trainer.train()

# -------------------------------
# EVALUATION
# -------------------------------
print("\n🔍 Evaluation on Test Set:")
predictions, labels, _ = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis=2)

true_predictions = [
    [ID2LABEL[p] for (p, l) in zip(pred, lab) if l != -100]
    for pred, lab in zip(predictions, labels)
]
true_labels = [
    [ID2LABEL[l] for (p, l) in zip(pred, lab) if l != -100]
    for pred, lab in zip(predictions, labels)
]

print(classification_report(true_labels, true_predictions))

# -------------------------------
# SAVE MODEL
# -------------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ BioBERT NER model saved to: {OUTPUT_DIR}")
