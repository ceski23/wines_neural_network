import matplotlib.pyplot as plt
import random
 
x, y = [0], [0]
 
plt.show()
plt.grid()
 
axes = plt.gca()
axes.set_ylim(0, 100)
line, = axes.plot(x, y, 'r-')
 
for i in range(200):
    new_y = (y[-1] + random.randint(0, 5)) % 101
    x.append(i)
    y.append(new_y)

    line.set_xdata(x)
    line.set_ydata(y)

    axes.set_xlim(0, i)

    plt.draw()
    plt.pause(1e-17)

plt.show()