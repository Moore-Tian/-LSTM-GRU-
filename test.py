import matplotlib.pyplot as plt


loss_values = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0]
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')

plt.show()