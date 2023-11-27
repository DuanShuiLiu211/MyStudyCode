import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

plt.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"

fig, ax = plt.subplots()
ax.set_xlim(-0.1, 2 * np.pi + 0.1)
ax.set_ylim(-1.1, 1.1)
(ln,) = plt.plot([], [], "-")

x = np.linspace(0, 2 * np.pi, 1000)


def update(frame):
    y = frame * np.sin(x)
    ln.set_data(x, y)
    return (ln,)


ani = FuncAnimation(fig, update, frames=1000, interval=1000 / 144)

ani.save("lol.gif")
