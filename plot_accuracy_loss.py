import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

log_path = './logs/'

ea = event_accumulator.EventAccumulator(log_path, size_guidance={
    event_accumulator.SCALARS: 0,
})
ea.Reload()

# Fetch losses
eval_losses = ea.Scalars('eval/loss')
train_losses = ea.Scalars('train/loss')

# Fetch accuracies
eval_accuracies = ea.Scalars('eval/accuracy')
train_accuracies = ea.Scalars('train/train_accuracy')

# Extract steps and values for losses
eval_steps = [e.step for e in eval_losses]
eval_loss_values = [e.value for e in eval_losses]
train_steps = [e.step for e in train_losses]
train_loss_values = [e.value for e in train_losses]

# Extract steps and values for accuracies
eval_accuracy_values = [e.value for e in eval_accuracies]
train_accuracy_values = [e.value for e in train_accuracies]

# #total_data_ * training_fraction / #batches
steps_per_epoch = 18000*0.8/8

# Steps to epochs
eval_epochs1 = [step / steps_per_epoch for step in eval_steps]
eval_epochs2 = [step / steps_per_epoch for step in train_steps]


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(eval_epochs2, train_loss_values, label='Train', color='Blue')
plt.plot(eval_epochs1, eval_loss_values, label='Validation', color='Coral')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(eval_epochs1, eval_accuracy_values, label='Validation', color='Coral')
plt.plot(eval_epochs1, train_accuracy_values, label='Train', color='Blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()