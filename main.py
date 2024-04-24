train_df = pd.read_csv("training.csv",
                       names=["id", "entity", "sentiment", "text"])
train_df.head()
test_df = pd.read_csv("validation.csv", 
                     names=["id", "entity", "sentiment", "text"])

train_df.isna().sum()
test_df.isna().sum()
train_df = train_df.dropna()
# Data size
print(f"Training set: {len(train_df)} rows")
print(f"Test set: {len(test_df)} rows")
# Distribution of the text length
sns.histplot(train_df["text"].str.len(), binwidth=50)
plt.show()
# Sample  from each class
sample_size = 3
train_df.groupby('sentiment').apply(lambda x: x.sample(sample_size))
# Create a mapping of labels to numbers
labels = list(train_df['sentiment'].unique())
id2label = {k:v for k,v in enumerate(labels)}
label2id = {v:k for k,v in enumerate(labels)}
print(labels)
print(id2label)
print(label2id)
train_df['label'] = train_df['sentiment'].map(label2id)
test_df['label'] = test_df['sentiment'].map(label2id)

train_df.head()
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
train = train_df.sample(frac=0.9, random_state=42)
valid = train_df.drop(train.index)
train
from datasets import Dataset

train_ds = Dataset.from_pandas(train)
valid_ds = Dataset.from_pandas(valid)
test_ds = Dataset.from_pandas(test_df)

train_ds, valid_ds, test_ds
train_ds['text'][0]
# Dynamic Padding
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
from datasets import load_metric

accuracy = load_metric('accuracy')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

train_ds = train_ds.map(preprocess_function, batched=True)
valid_ds = valid_ds.map(preprocess_function, batched=True)
test_ds = test_ds.map(preprocess_function, batched=True)
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(labels), id2label=id2label, label2id=label2id, 
    ignore_mismatched_sizes=True, # The model was pretrained with 3 labels, but our dataset has 4 labels (a new head with random weights will be initialized)
)
# Adapted from https://huggingface.co/docs/transformers/tasks/sequence_classification
training_args = TrainingArguments(
    output_dir="DETECTOR",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
predictions = trainer.predict(test_ds)
preds = np.argmax(predictions.predictions, axis=-1)
accuracy.compute(predictions=preds, references=predictions.label_ids)
texts = [
    "I am good. I am scared about tommorow"
]
inputs = tokenizer(texts, return_tensors="pt", padding=True)
model = model.to('cpu') # put model to cpu
logits = model(**inputs).logits
print([model.config.id2label[item.item()] for item in logits.argmax(axis=-1)])
from tabulate import tabulate 
Evaluation_metrics_table()
Evaluation_metrics_accuracy()
Evaluation_metrics()
Evaluation_metrics_train_loss()
Evaluation_metrics_validation_loss()