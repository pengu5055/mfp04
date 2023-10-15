import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import time
from scipy.optimize import curve_fit
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


def dft_inverse(x):
    a = np.copy(x)
    output = []
    N = a.shape[0]
    for k in range(N):
        f_k = 0
        for n in range(N):
            f_k += 1/N*(a[n] * np.exp(2j * np.pi * k * n / N))
        output.append(f_k)

    return output


def prepare_time(n, T):
    t_min = 0
    t_max = T
    return np.linspace(t_min, t_max, n, endpoint=False)


def Gauss_norm(x, mu, sig):
    return 1/(np.sqrt(2*np.pi)*sig) * np.exp((x - mu)**2 / (2*sig**2))


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
        freq[:N] = np.arange(0, N)
        freq[N:] = np.arange(-N - 1, 0)

    return freq*scale


def time_dft(signal, n, T, r):
    """
    Function times dft(x) for different values of n. Runs r tries. Returns array of time averages.

    signal: signal function to calculate dft(signal)
    n: ARRAY of different sample sizes
    T: sample spacing
    r: number of repeats
    """
    output = []
    for k in range(r):
        times = []
        for element in list(n):
            t = prepare_time(element, T)
            sig = signal(t)
            print(element)
            pre = time.time()
            ft_signal = dft(sig)
            post = time.time()
            times.append(post-pre)  # Should return seconds elapsed
        output.append(times)

    return np.average(output, axis=0)


def sine_cosine(t):
    return np.sin(2*np.pi*t/100) + np.cos(2*np.pi*t/50)


def square(x, a, b, c):
    return a*x**2 + b*x + c

n = 200
T = 1000
nyq_freq = 1/(2*T)
print("Samples: {}\nSample spacing: {}\nNyquist Frequency: {}\n".format(n, T, nyq_freq))
t = prepare_time(n, T)
nu = prepare_freq(n, T)

# Simple signal plot

# signal = np.sin(2*np.pi*t/100) + np.cos(2*np.pi*t/50)
# signal = np.sin(2*np.pi*t/100) + np.cos(2*np.pi*t/34)
# # signal = np.exp(-t/50)*np.sin(t)
# nu_y = dft(signal)
# nu = np.roll(nu, 99)  # Lepsi ploti
# nu_y = np.roll(nu_y, 99)
#
# plt.plot(t, signal, label=r"$ \sin{(\frac{2\pi t}{100}}) + \cos{(\frac{2\pi t}{34})}$", color="#7F2982")
# # plt.plot(t, signal, label=r"$ e^{-t/50}\cdot \sin(t)$", color="#7F2982")
# plt.scatter(t, signal, label=r"Vzorčenje pri $n={}$ in $T={}$".format(n, T), color="#DE639A", s=5)
# plt.title("Vzorčenje signala")
# plt.legend(loc="lower right")
# plt.show()
#
# fig, (ax1, ax2, ax3) = plt.subplots(3)
# fig.set_figheight(6)
# fig.set_figwidth(6)
# for ax in fig.get_axes():
#     ax.label_outer()
# ax1.plot(nu, np.real(nu_y), color="#9797EE")
# ax2.plot(nu, np.imag(nu_y), color="#FCB07E")
# ax3.plot(nu, np.abs(nu_y)**2, color="#94C9A9")
# plt.suptitle(r"FT $\sin{(\frac{2\pi t}{100})} + \cos{(\frac{2\pi t}{34})}$")
# # plt.suptitle(r"FT $e^{-t/50}\cdot \sin(t)$")
# ax1.set_title("Realni del")
# ax2.set_title("Imaginarni del")
# ax3.set_title("Kompleksni kvadrat")
# ax3.sharex = True
# plt.xlabel(r"$\nu$")
# plt.show()

# Addon: Signal FT np.fft.fft comparison
# signal = np.sin(2*np.pi*t/100) + np.cos(2*np.pi*t/50)
#
# nu_y2 = np.fft.fft(signal)
# nu_y2 = np.roll(nu_y2, 99)
# plt.plot(nu, nu_y2)
# plt.show()
# plt.plot(nu, np.abs(nu_y - nu_y2), color="#907AD6")
# plt.scatter(nu, np.abs(nu_y - nu_y2), color="#7FDEFF", s=5)
# plt.title(r"Absolutna razlika $\mathrm{dft}(s(t))$ in $\mathrm{np.fft.fft}(s(t))}$")
# plt.xlabel(r"$\nu$")
# plt.ylabel(r"$|dft(s(t)) - np.fft.fft(s(t))|$")
# plt.yscale("log")
# plt.show()

# Inverse FT plot (n = 2000, T = 1000 for all)

# signal = np.sin(2*np.pi*t/100) + np.cos(2*np.pi*t/50)
# signal = np.sin(2*np.pi*t/100)/np.cos(2*np.pi*t/34.5607895)
# signal = np.exp(-t/50)*np.sin(t)
# signal = np.sin(2*np.pi*t/100) + np.cos(2*np.pi*t/0.08)
# ft_signal = dft(signal)
# reconstructed_signal = dft_inverse(ft_signal)
# x = np.linspace(1, 1000, 5000)
# plt.plot(x, np.sin(2*np.pi*x/100) + np.cos(2*np.pi*x/0.08))
# plt.plot(t, signal, color="#DE639A")
# plt.show()
# fig, (ax1, ax2, ax3) = plt.subplots(3)
# fig.set_figheight(6)
# fig.set_figwidth(6)
# for ax in fig.get_axes():
#     ax.label_outer()
# # ax1.plot(t, signal, color="#9797EE", label="Signal")
# ax1.plot(x, np.sin(2*np.pi*x/100) + np.cos(2*np.pi*x/0.1))
# # ax1.scatter(t, signal, color="#DE639A", label=r"Vzorčenje pri $n={}$ in $T={}$".format(n, T), s=5)
# ax2.plot(np.roll(nu, int(n/2) - 1), np.roll(np.abs(ft_signal)**2, int(n/2) - 1), color="#94C9A9", label="FT signala")
# ax3.plot(t, reconstructed_signal, color="#FCB07E", label="Rekonstrukcija signala")
# ax1.set_title("Originalen signal")
# ax2.set_title("FT signala")
# ax3.set_title("Rekonstrukcija signala")
# # # ax1.legend()
# plt.suptitle("Rekonstrukcija signala z inverzno Fourierovo transformacijo")
#
# plt.show()

# plt.plot(t, np.abs(signal-reconstructed_signal))
# plt.yscale("log")
# plt.show()

# Plots and code for Gauss
# x = np.linspace(-3, 3, 200)
#
# plt.plot(x, np.roll(scipy.stats.norm.pdf(x, loc=0, scale=0.5), 100), color="#772D8B")
# plt.title(r"Periodična Gaussova porazdelitev z $\mu = 0$ in $\sigma=0.5$")
# plt.xlabel(r"$x$")
# plt.ylabel(r"$f(x)$")
# plt.show()
#
# signal = scipy.stats.norm.pdf(x, loc=0, scale=0.1)
# nu_y = dft(signal)
# nu_y = np.roll(nu_y, int(n/2) - 1)  # Rocno rolled arraya
# nu = np.roll(nu, int(n/2) - 1)
# plt.plot(x, signal, color="#7F2982", label="Gaussova funkcija")
# plt.scatter(x, signal, color="#DE639A", label="Vzorci $n = 200$ in $T = 100$")
# plt.title("Vzorčenje Gaussove funkcije")
# plt.xlabel(r"$x$")
# plt.ylabel(r"$f(x)$")
# plt.legend()
# plt.show()
#
# fig, (ax1, ax2) = plt.subplots(2)
# fig.set_figheight(6)
# fig.set_figwidth(6)
# for ax in fig.get_axes():
#     ax.label_outer()
# ax1.plot(nu, np.real(nu_y * np.exp(2*np.pi*1j*nu*T*(n/2))), color="#9797EE", label="Popravljena")
# ax1.plot(nu, np.real(nu_y), color="#94C9A9", label="Nepopravljena")
# ax2.plot(nu, np.imag(nu_y * np.exp(2*np.pi*1j*nu*T*(n/2))), color="#FCB07E")
# plt.suptitle("Fourierova transformacija Gaussove porazdelitve")
# ax1.set_title("Realni del")
# ax2.set_title("Imaginarni del")
# ax1.legend()
# plt.show()

# Animation of Nyquist frequency

fig, ax = plt.subplots()
x = np.linspace(-3, 3, n, endpoint=False)

img_re = []
img_im = []
img_sq = []
for i in range(1, 3000, 10):
    signal = np.sin(2 * np.pi * t / 100) + np.cos(2 * np.pi * t / i)
    nu_y = dft(signal)
    nu = np.delete(nu, np.s_[int(n // 2):])
    nu_y = np.delete(nu_y, np.s_[int(n // 2):])
    img_sq.append(plt.plot(nu, np.abs(nu_y)**2, color="#94C9A9"))
    print(i)

ani = ArtistAnimation(fig, img_sq, interval=30, repeat=False, blit=False)


# plt.scatter(t, signal, label=r"Vzorčenje pri $n={}$ in $T={}$".format(n, T), color="#DE639A", s=5)
plt.title("Animacija prečkanja Nyquistove frekvence")
plt.xlabel(r"$x$")
plt.ylabel(r"$FT[f(x)]$")
ani.save("nyquist_1_4000.mp4", "ffmpeg", fps=30)
plt.show()

# Time dependency of ( dft(x) )(n)
# fig, ax = plt.subplots()
# test_n = np.arange(1, 1000)
# times = time_dft(sine_cosine, test_n, T, 1)
# plt.scatter(test_n, times, color="#92D5E6", label="Measured", s=6)
# # plt.scatter(test_n, times, color="#7FDEFF")
# fitpar, fitcov = curve_fit(square, xdata=test_n, ydata=times)
# yfit = square(test_n, fitpar[0], fitpar[1], fitpar[2])
# fittext= "Quadratic fit: $y = ax^2 + bx+ c$\na = {} ± {}\nb = {} ± {}\nc = {} ± {}".format(format(fitpar[0], ".4e"), format(fitcov[0][0]**0.5, ".4e"),
#                                                                                            format(fitpar[1], ".4e"), format(fitcov[1][1]**0.5, ".4e"),
#                                                                                            format(fitpar[2], ".4e"), format(fitcov[2][2]**0.5, ".4e"))
# plt.text(0.5, 0.12, fittext, ha="left", va="center", size=10, transform=ax.transAxes, bbox=dict(facecolor="#a9f5ee", alpha=0.5))
# plt.plot(test_n, yfit, color="#772D8B", label="Quadratic fit")
# plt.title("Odvisnost časa računanja DFT od števila vzorcev")
# plt.xlabel(r"$n$")
# plt.ylabel(r"$t$ [s]")
# plt.legend()
# plt.show()

