import json
import torch
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # GPU!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

def predict_probabilities(model, tokenizer, article, device):
    inputs = tokenizer(article, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # GPU!
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probabilities.squeeze().cpu().tolist()

model_path = "article_bias_classifier"
model, tokenizer, device = load_model(model_path)

# Unseen data
with open('unseen_filtered_data.json', 'r') as f:
    unseen_data = json.load(f)

predicted_probs = []
actual_labels = []

# tqdm
for item in tqdm(unseen_data, desc="Predicting biases"):
    content = item['content']
    actual_bias = item['bias']
    probs = predict_probabilities(model, tokenizer, content, device)
    predicted_probs.append(probs)
    actual_labels.append(actual_bias)

# Lists to numpy arrays
predicted_probs = np.array(predicted_probs)
actual_labels = np.array(actual_labels)
actual_labels_binarized = label_binarize(actual_labels, classes=[0, 1, 2])

# ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}
mean_fpr = np.linspace(0, 1, 100)
tprs = []
for i, label in enumerate(['Left', 'Center', 'Right']):
    fpr[i], tpr[i], _ = roc_curve(actual_labels_binarized[:, i], predicted_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    tprs.append(np.interp(mean_fpr, fpr[i], tpr[i]))
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)


# ROC plot
sns.set()
plt.figure(figsize=(10, 8))

for i, label in enumerate(['Left', 'Center', 'Right']):
    plt.plot(fpr[i], tpr[i], linewidth=2, label=f'{label} (AUC = {roc_auc[i]:.2f})')
    
plt.plot(mean_fpr, mean_tpr, color='black', linestyle='--', linewidth=2, label=f'Average (AUC = {mean_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random Guess', linewidth=1.5)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('One-vs-Rest ROC Curves for Unseen Test Data', fontsize=16)
plt.legend(title="Bias Label", title_fontsize='13', fontsize='12', loc="lower right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# Confusion matrix
predicted_labels = np.argmax(predicted_probs, axis=1)
cm = confusion_matrix(actual_labels, predicted_labels)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm_percentage, cmap='Blues', annot=False,
                 xticklabels=['Left', 'Center', 'Right'],
                 yticklabels=['Left', 'Center', 'Right'])

threshold = cm_percentage.max() / 2
for i in range(cm_percentage.shape[0]):
    for j in range(cm_percentage.shape[1]):
        color = "white" if cm_percentage[i, j] > threshold else "black"
        ax.text(j + 0.5, i + 0.5, f'{cm_percentage[i, j]:.1f}%', 
                horizontalalignment='center', 
                verticalalignment='center', 
                color=color, fontsize=13)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Confusion Matrix for Unseen Test Data', fontsize=16)
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.show()
