import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

plt.plot(x, y)
plt.show()

""" epochs = []
loss_values = []

with open('train.log', 'r') as file:
    for line in file:
        epoch_loss = line.strip().split(' ')
        epoch = int(epoch_loss[1])
        loss = float(epoch_loss[4])
        epochs.append(epoch)
        loss_values.append(loss)

epochs = [i for i in range(1, len(loss_values) + 1)]
loss_values = [i for i in epochs]

plt.plot(epochs, loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.savefig('loss_vs_epoch.png') """