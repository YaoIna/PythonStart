import numpy as np
import matplotlib.pyplot as plt


def plot_mel_and_hz():
    x_hz = np.arange(20, 10010, 10)
    x_hz = x_hz.reshape(x_hz.shape[0], -1)
    y_mel = 2595 * np.log10(1 + x_hz / 700)

    plt.xscale('log')
    plt.plot(x_hz, y_mel)
    caption = "Relationship between Mel and Hz"
    plt.title('Mel vs. Hz')
    plt.xlabel('Hertz Scale\n\n%s' % caption)
    plt.ylabel('Mel Scale')
    plt.grid(True)
    plt.show()
