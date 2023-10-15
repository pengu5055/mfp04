import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from matplotlib.animation import ArtistAnimation


def dft(x):
    """
    x: 1D array of f(x) to compute F(nu)
    n: number of data points
    T: sampling period
    dt = T/n
    1/dt: sampling frequency
    0.5/dt: critical frequency

    Returns array F(nu)
    """
    a = np.copy(x)
    output = []
    for k in range(a.shape[0]):
        f_k = 0
        for n in range(a.shape[0]):
            f_k += a[n] * np.exp(-2j * np.pi * k * n/a.shape[0])
        output.append(f_k)
    return output


def prepare_time(n, T):
    t_min = 0
    t_max = T
    return np.linspace(t_min, t_max, n, endpoint=False)


def prepare_freq(n, T):
    """
    Return DFT frequencies.
    n: number of samples (window size?)
    T: sample spacing (1/sample_rate)
    """
    freq = np.empty(n)
    scale = 1/(n * T)

    if n % 2 == 0:  # Even
        N = int(n/2 + 1)
        freq[:N] = np.arange(0, N)
        freq[N:] = np.arange(-(n/2) + 1, 0)
    else:
        N = int((n-1)/2)
        print(N)
        freq[:N] = np.arange(0, N)
        freq[N:] = np.arange(-N - 1, 0)

    return freq*scale


def load_wav(filename, preparetime=False):
    fs_rate, signal = wavfile.read(filename)
    if len(signal.shape) == 2:
        signal = signal.sum(axis=1)/2  # Povpreci oba channela
    n = signal.shape[0]  # Number of samples
    secs = n / fs_rate
    T = 1/fs_rate
    print("Sampling frequency: {}\nSample period: {}\nSamples: {}\nSecs: {}\n".format(fs_rate, T, n, secs))
    if preparetime:
        # t = np.linspace(0, secs, n)
        t = np.arange(0, secs, T)
        return signal, t
    else:
        return signal


def analyze_wav(filename, onesided=False):
    fs_rate, signal = wavfile.read(filename)
    if len(signal.shape) == 2:
        signal = signal.sum(axis=1)/2  # Povpreci oba channela
    n = signal.shape[0]  # Number of samples
    secs = n / fs_rate
    T = 1/fs_rate
    print("Sampling frequency: {}\nSample period: {}\nSamples: {}\nSecs: {}\n".format(fs_rate, T, n, secs))
    # ft = dft(signal) # waaaaaaay slow
    ft = np.fft.fft(signal)
    # t = np.linspace(0, secs, n)
    t = np.arange(0, secs, T)
    if onesided:
        return t, signal, prepare_freq(np.array(ft).size//2, t[1]-t[0]), ft[:n//2]
    return t, signal, prepare_freq(np.array(ft).size, t[1]-t[0]), ft, n


# data, t = load_wav('sine_440.wav', preparetime=True)
# dft = np.fft.fft(data)
# freqs = np.fft.fftfreq(data.size, t[1]-t[0])

fig, ax = plt.subplots()
filenames = ["Bach.44100.wav", "Bach.11025.wav", "Bach.5512.wav", "Bach.2756.wav", "Bach.1378.wav", "Bach.882.wav"]
img = []


for filename in filenames:
    t, signal, freq, ft_signal, n = analyze_wav(filename)
    print(ft_signal)
    freq = np.delete(freq, np.s_[int(n//2):])
    ft_signal = np.delete(ft_signal, np.s_[int(n//2):])
    print(ft_signal)
    img.append(plt.plot(freq, np.abs(ft_signal)**2, color="#772D8B"))
    plt.title("Spekter datoteke {}".format(filename))
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"Amplituda")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(0, 25000)
    # plt.show()
plt.title("Degredacija kvalitete Bachove partite")
ani = ArtistAnimation(fig, img, interval=1000, repeat=True, blit=True)
ani.save("bach_degredation.mp4", "ffmpeg", fps=2)
# plt.plot(np.fft.fftfreq(np.array(signal).size, t[1]-t[0]), np.abs(np.fft.fft(signal))**2)
plt.show()
