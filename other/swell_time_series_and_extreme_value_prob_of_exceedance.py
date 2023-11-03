import numpy as np
import matplotlib.pyplot as plt


T = 6000*1
fs = 100  # sample frequency
n_samples = int(T * fs)
# f_arr = np.arange(1 / T, fs / 2, 1 / T)
f_arr_w_zero = np.fft.rfftfreq(n_samples, 1/fs)  # with the zero frequency included
f_arr = f_arr_w_zero[1:]
n_freq = len(f_arr)

w_arr = 2*np.pi * f_arr  # omega array

# Jonswap spectrum according to DNVGL-RP-C205 (2019), 3.5.5.2
Tp = 14.2  # Enclosure 2 NOT-356 Parametric Excitation, 3.5.2
Hs = 0.34  # Enclosure 2 NOT-356 Parametric Excitation, 3.5.2
gamma = 5.0  # Enclosure 2 NOT-356 Parametric Excitation, 3.5.2
A_gama = 0.2 / (0.065 * gamma**0.803 + 0.135)
w_p = 2*np.pi / Tp  # angular spectral peak frequency
sigma_a = 0.07  # Metocean Specification MOS_Bjørnafjorden_rev2E, 2.8.1
sigma_b = 0.09  # Metocean Specification MOS_Bjørnafjorden_rev2E, 2.8.1
sigma_arr = np.array([sigma_a if i <= w_p else sigma_b for i in w_arr])
PSD_PM = 5/16 * Hs**2 * w_p**4 * w_arr**-5 * np.exp(-5/4 * (w_arr/w_p)**-4)  # Pierson-Moskowitz spectrum
PSD_J_omegas = A_gama * PSD_PM * gamma ** np.exp(-0.5 * ((w_arr - w_p)/(sigma_arr * w_p))**2)  # Jonswap Power spectrum
PSD_J = PSD_J_omegas * 2*np.pi  # because S(f)*delta_f = S(w)*delta_w. See eq. (2.68) from Strommen.

# To obtain the time series, the IFFT should be applied on the ASD (Amplitude spectral density) and not the PSD. ASD = sqrt(PSD). Note that the FFT on the time series gives an ASD and not PSD (the ASD)must be squared).
ASD_J = np.sqrt(PSD_J) * np.exp(1j * np.random.uniform(0, 2 * np.pi, n_freq))
ASD_J = np.insert(ASD_J, 0, 0)

signal_J = np.fft.irfft(ASD_J) * np.sqrt(n_samples)  # * np.sqrt(n_samples) / np.sqrt(2)
ASD_J_reconstructed = np.fft.rfft(signal_J)
print("Good reconstruction" if np.allclose(ASD_J**2, ASD_J_reconstructed**2) else "Bad reconstruction")

t_array = np.linspace(0.0, T, n_samples, endpoint=False)

plt.plot(t_array, np.real(signal_J), label='x1')
plt.legend()

plt.close()

print( 4* np.std(np.real(signal_J)))






























###############################    ONLY TRASH BELOW THIS LINE    #################################
#
# plt.plot(f_arr, PSD_J, alpha=0.4, label='S_J')
# plt.plot(f_arr, PSD_J_reconstructed, alpha=0.4, label='S_J_new')
# plt.legend()
#
#
#
#
#
#
# # Verify that P matches P_new
# f_new, S_J_new = scipy.signal.periodogram(x_new, detrend=False)
#
#
#
#
#
# # # Plot Jonswap spectrum
# # plt.plot(f_arr, S_J)
# # plt.show()
#
# # From PSD to time-series. Source: https://dsp.stackexchange.com/questions/83744/how-to-generate-time-series-from-a-given-one-sided-psd/83767#83767?newreg=0a9aef025f81412dadfa4ed45df7af70
# N = 2*(n_freq - 1)
# # Because S_J includes both DC and Nyquist (N/2+1), P_fft must have 2*(n_freq-1) elements
# S_J[0] = S_J[0] * 2
# S_J[-1] = S_J[-1] * 2
# P_fft_new = np.zeros((N,), dtype=complex)
# P_fft_new[0:int(N/2)+1] = S_J
# P_fft_new[int(N/2)+1:] = S_J[-2:0:-1]
# X_new = np.sqrt(P_fft_new)
# # Create random phases for all FFT terms other than DC and Nyquist
# phases = np.random.uniform(0, 2*np.pi, (int(N/2),))
# # Ensure X_new has complex conjugate symmetry
# X_new[1:int(N/2)+1] = X_new[1:int(N/2)+1] * np.exp(2j*phases)
# X_new[int(N/2):] = X_new[int(N/2):] * np.exp(-2j*phases[::-1])
# X_new = X_new * np.sqrt(n_freq) / np.sqrt(2)
# # This is the new time series with a given PSD
# x_new = np.real(scipy.fft.ifft(X_new))
#
#
#
#
# f, P = scipy.signal.periodogram(x_new, 1, 'boxcar', nfft=n_freq, detrend=False)
#
# plt.plot(f_arr, S_J, label='S_J')
# plt.plot(f, P, label='S_J_reconstructed')
# plt.legend()
# plt.show()
#
#
#
#
# # plt.plot(x_new)
#
# # # TRASH FAILED ATTEMPT: From PSD to time-series
# # dt = 0.1  # s
# # A = np.zeros(n_freq, dtype='complex')
# # for w_idx in range(n_freq):
# #     S = S_J[w_idx]
# #     L, D, _ = scipy.linalg.ldl(S, lower=True)
# #     G = L @ np.sqrt(D)
# #     A[w_idx] = G * np.exp(1j * 2 * np.pi * np.random.random())
# # Nu = np.block([[np.zeros(1)], [A[:n_freq]], [np.real(A[n_freq-1])], [np.conj(np.flipud(A[:n_freq]))]])
# # speed = np.real(np.fft.ifft(Nu, axis=0) * np.sqrt(n_freq / dt))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import numpy as np
# import scipy
# import matplotlib.pyplot as plt
#
#
#
#
# N = 1000
#
# # Generate white noise and filter it to give it an interesting PSD
# t = np.linspace(0, 600, N)
# x = np.cos(2*np.pi/14 * t)
#
# # Find PSD using built-in function and manually:
# # Built-in function
# f, P = scipy.signal.periodogram(x, detrend=False)
#
# # # Manually
# # N = len(x)
# # X = scipy.fft.fft(x) / np.sqrt(N) * np.sqrt(2)
# # P_fft = np.abs(X*np.conj(X))
# # P_fft_one_sided = P_fft[0:int(N/2)+1]  # P_fft_one_sided is identical to P
# # P_fft_one_sided[0] = P_fft_one_sided[0] / 2
# # P_fft_one_sided[-1] = P_fft_one_sided[-1] / 2
# # print(np.allclose(P, P_fft_one_sided))  # True, P matches P_fft_one_sided
# #
#
# # Now undo the manual operation of P_fft_one_sided to get back to a time series, insert your PSD here
# N_P = len(P)  # Length of PSD
#
# # Because P includes both DC and Nyquist (N/2+1), P_fft must have 2*(N_P-1) elements
# P[0] = P[0] * 2
# P[-1] = P[-1] * 2
# P_fft_new = np.zeros((N,), dtype=complex)
# P_fft_new[0:int(N/2)+1] = P
# P_fft_new[int(N/2)+1:] = P[-2:0:-1]
#
# X_new = np.sqrt(P_fft_new)
#
# # Create random phases for all FFT terms other than DC and Nyquist
# phases = np.random.uniform(0, 2*np.pi, (int(N/2),))
#
# # Ensure X_new has complex conjugate symmetry
# X_new[1:int(N/2)+1] = X_new[1:int(N/2)+1] * np.exp(2j*phases)
# X_new[int(N/2):] = X_new[int(N/2):] * np.exp(-2j*phases[::-1])
# X_new = X_new * np.sqrt(N) / np.sqrt(2)
#
# # This is the new time series with a given PSD
# x_new = np.real(scipy.fft.ifft(X_new))
#
# # Verify that P matches P_new
# f_new, P_new = scipy.signal.periodogram(x_new, detrend=False)
# print(np.isclose(P, P_new))  # True, P matches P_new
#
#
# plt.plot(t, x, label='x')
# plt.plot(t, x_new, label='x_new')
#
#
#
#
