from cProfile import label
import re 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
avg_train_loss = []
avg_val_loss = []
avg_train_acc = []
avg_val_acc = []

with open('finetune_log.txt') as f:
    logs = f.readlines()
    logs = [i[:-1] for i in logs]

for i in logs:
    train_loss = re.findall(r"avg train loss: (\d+.\d+)",i)
    train_acc = re.findall(r"avg train acc: (\d+.\d+)",i)
    val_loss = re.findall(r"avg val loss: (\d+.\d+)",i)
    val_acc = re.findall(r"avg val acc: (\d+.\d+)",i)

    if train_loss:  avg_train_loss.append(float(train_loss[0]))
    if train_acc:  avg_train_acc.append(float(train_acc[0]))
    if val_loss:  avg_val_loss.append(float(val_loss[0]))
    if val_acc:  avg_val_acc.append(float(val_acc[0]))


epoch = [i for i in range(19)]
# plt.plot(epoch, avg_train_loss, label='train')
# plt.plot(epoch, avg_val_loss, label='val')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

plt.plot(epoch, avg_train_acc, label='train')
plt.plot(epoch, avg_val_acc, label='val')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()



