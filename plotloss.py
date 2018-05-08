# -*- coding: utf-8 -*-
import csv
import matplotlib.pyplot as plt

# Plot loss, accuracy, and learning rates for a training session

epochs = []
losses = []
accs = []
lrs = []
with open('att3.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        print(row[0])
        if row[0] == 'epoch':
            break
        epoch = float(row[0])
        batch = float(row[1])/390.0
        time = epoch + batch
        loss = float(row[2])
        acc = float(row[3])
        lr = float(row[4])
        epochs.append(time)
        losses.append(loss)
        accs.append(acc)
        lrs.append(lr)

fig = plt.figure()
plt.plot(epochs, accs)
fig.suptitle('ATT3 accuracy', fontsize=20)
plt.xlabel('epochs', fontsize=18)
plt.ylabel('accuracy', fontsize=16)
plt.show()
