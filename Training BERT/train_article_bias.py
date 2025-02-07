import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from copy import deepcopy

data = pd.read_json('filtered_data.json')
data['bias'] = data['bias'].astype(int)

class ArticleDataset(Dataset):
    def __init__(self, article, biases, tokenizer):
        self.encodings = tokenizer(article, truncation=True, padding=True, max_length=512)
        self.labels = biases

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_texts, val_texts, train_labels, val_labels = train_test_split(data['content'], data['bias'], test_size=0.2)
train_dataset = ArticleDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = ArticleDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

training_args = TrainingArguments(
    output_dir='article_bias_classifier',
    num_train_epochs=20,
    per_device_train_batch_size=8,  # Def: 8
    per_device_eval_batch_size=8,   # Def: 8
    warmup_steps=500,
    
    weight_decay=0.04,
    learning_rate=5e-5,
    lr_scheduler_type='linear',  # Better with lr schedualr

    logging_dir='./logs',
    logging_steps=180,  # 1800 steps per epoch, so 180->10logs/epoch
    fp16=True,          # Using both 16 and 32 bit precisino during training to speed up
    
    evaluation_strategy="steps",
    save_strategy="steps",  # Must equal to evaluation_strategy
    save_steps=9000,        # Save model every 5 epochs (1800*5 steps)
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.eval_steps == 0:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.add_callback(CustomCallback(trainer))

trainer.train()
model_path = "article_bias_classifier"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)