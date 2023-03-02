import json
import matplotlib.pyplot as plt

# # Define data values
# x = [7, 14, 21, 28, 35, 42, 49]
# y = [5, 12, 19, 21, 31, 27, 35]
# z = [3, 5, 11, 20, 15, 29, 31]

# # Plot a simple line chart
# plt.plot(x, y)

# # Plot another line on the same chart/graph
# plt.plot(x, z)

# plt.show()
f = open('log.json', 'r')

data = json.load(f)

iters = []
loss_pixs = []
loss_cleans = []

for i in data['json']:
    iters.append(i['iter'])
    loss_pixs.append(i['loss_pix'])
    loss_cleans.append(i['loss_clean'])

plt.plot(iters, loss_pixs)
plt.show()

f.close()