import matplotlib.pyplot as plt


epochs = []
loss_values = []

with open("log/log.txt", 'r') as file:
    for line in file:
        epoch_loss = line.strip().split(' ')
        epoch = int(epoch_loss[1])
        loss = float(epoch_loss[4])
        epochs.append(epoch)
        loss_values.append(loss)

plt.plot(epochs, loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.savefig('loss_vs_epoch.png')
plt.show()
