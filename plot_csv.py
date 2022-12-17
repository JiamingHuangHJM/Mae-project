from cProfile import label
import csv
import matplotlib.pyplot as plt


with open('cls_acc_train-sc.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    cls_acc_train = [line for line in csv_reader]

cls_acc_train = [i[2] for i in cls_acc_train][1:]


with open('cls_acc_val-sc.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    cls_acc_val = [line for line in csv_reader]

cls_acc_val = [i[2] for i in cls_acc_val][1:]


with open('cls_loss_train-sc.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    cls_loss_train = [line for line in csv_reader]

cls_loss_train = [i[2] for i in cls_loss_train][1:]


with open('cls_loss_val-sc.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    cls_loss_val = [line for line in csv_reader]

cls_loss_val = [i[2] for i in cls_loss_val][1:]


plt.plot([i for i in range(len(cls_acc_train))], [float(i) for i in cls_acc_train], label="train acc")
plt.plot([i for i in range(len(cls_acc_val))], [float(i) for i in cls_acc_val], label="val acc")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
plt.clf()

plt.plot([i for i in range(len(cls_acc_train))], [float(i) for i in cls_loss_train], label="train loss")
plt.plot([i for i in range(len(cls_acc_val))], [float(i) for i in cls_loss_val], label="val loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()



