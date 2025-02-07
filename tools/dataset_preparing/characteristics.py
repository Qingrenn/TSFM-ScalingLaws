import numpy as np
import re
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from pycatch22.catch22 import catch22_all
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq


def z_score(x: np.ndarray):
    assert x.ndim == 1
    mean, std = np.mean(x), np.std(x)
    z = (x - mean) / (std + 1e-5)
    return z


def freq_to_period(freq: str) -> int:
    """
    Convert a pandas frequency to a periodicity

    Parameters
    ----------
    freq : str or offset
        Frequency to convert

    Returns
    -------
    int
        Periodicity of freq

    Notes
    -----
    Annual maps to 1, quarterly maps to 4, monthly to 12, weekly to 52,
    daily to 7, business daily to 5, hourly to 24, minutely to 60, secondly to 60,
    microsecond to 1000, nanosecond to 1000
    """

    yearly_freqs = ("A-", "AS-", "Y-", "YS-", "YE-")
    if freq in ("A", "Y") or freq.startswith(yearly_freqs):
        return 1
    elif freq == "Q" or freq.startswith(("Q-", "QS", "QE")):
        return 4
    elif freq == "M" or freq.startswith(("M-", "MS", "ME")):
        return 12
    elif freq == "W" or freq.startswith("W-"):
        return 52
    elif freq == "D":
        return 7
    elif freq == "B":
        return 5
    elif freq == "H":
        return 24
    elif freq in ("T", "min"):
        return 60
    elif freq in ("S", "ms"):
        return 1000
    elif freq in ("L", "us"):
        return 1000
    else:  # pragma : no cover
        raise ValueError(
            "freq {} not understood. Please report if you "
            "think this is in error.".format(freq)
        )


def series_decomp(x: np.ndarray, period: int):
    assert x.ndim == 1
    stl = STL(x, period)
    res = stl.fit()
    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid
    return trend, seasonal, residual


def trend_and_seasonal_length(x: np.ndarray, period: int | str):
    assert x.ndim == 1

    if isinstance(period, str):
        match = re.match(r"(\d+)(\w+)", period)
        if match:
            freq_mag, freq_unit = match.groups()
            freq_mag = int(freq_mag)
            period = freq_to_period(freq_unit)

            point_per_period = period // freq_mag
            if point_per_period < 10:
                if freq_unit == "T":
                    period = int(point_per_period * 24)  # T -> D
                if freq_unit == "H":
                    period = int(point_per_period * 7)  # T -> W
        else:
            freq_mag = 1
            period = freq_to_period(period)

    x = impute(x)
    x = z_score(x)
    trend, seasonal, residual = series_decomp(x, period)
    trend_length = max(0, 1 - np.var(residual) / (np.var(x - seasonal) + 1e-5))
    seasonal_length = max(0, 1 - np.var(residual) / (np.var(x - trend) + 1e-5))
    return trend_length, seasonal_length, period


def stationarity(x: np.ndarray):
    """
    if p_value < 0.05:
        "Reject the null hypothesis (H0), the time series is stationary.
    else:
        "Fail to reject the null hypothesis (H0), the time series is non-stationary."
    """
    assert x.ndim == 1
    x = impute(x)
    x = z_score(x)

    if np.all(x == x[0]):
        return 0, True

    try:
        p_value = adfuller(x)[1]
    except Exception as e:
        p_value = None
        raise e
    finally:
        return p_value, p_value <= 0.05


def shifting(x: np.ndarray, bins: int = 5):
    assert x.ndim == 1
    x = impute(x)
    x = z_score(x)
    x_min, x_max = np.min(x), np.max(x)
    N = len(x)
    # statistics of each bin
    ms = list()
    for i in range(bins):
        lower_bound = x_min + i * (x_max - x_min) / bins
        k = np.where(x >= lower_bound)[0]
        m = (2 * np.median(k) - N) / N
        ms.append(m)
    return np.abs(np.median(ms))


def transition(x: np.ndarray):
    assert x.ndim == 1
    x = impute(x)
    x = z_score(x)
    tsfeatures = catch22_all(x)
    transition_variance = tsfeatures["values"][8]
    return transition_variance


def missing_rate(x: np.ndarray):
    assert x.ndim == 1
    missing_rate = np.sum(np.isnan(x)) / len(x)
    return missing_rate


def impute(x: np.ndarray):
    x = x.copy()
    x[np.isnan(x)] = np.nanmean(x)
    return x


def correlation(x: np.ndarray):
    assert x.ndim == 2

    tsfeatures = []
    for i in range(x.shape[0]):
        z = z_score(x[i])
        tsfeature = catch22_all(z)
        tsfeatures.append(tsfeature["values"])

    # calculate pearson correlation between tsfeature
    correlation = np.corrcoef(tsfeatures)
    correlation = np.mean(correlation) + 1 / (1 + np.var(correlation))

    return correlation


def calculate_snr(signal, noise):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def determine_cutoff(sample: np.ndarray):
    x = sample - np.mean(sample)
    fs = x.shape[0]
    X = fft(x)
    freqs = fftfreq(len(x), 1 / fs)
    magnitudes = np.abs(X)
    peak_freq = freqs[np.argmax(magnitudes)]
    cutoff = max(peak_freq, 1)
    return cutoff


def signal_noise_ratio(x: np.ndarray):
    assert x.ndim == 1
    x = impute(x)

    max_length = 10000
    if len(x) > max_length:
        x = x[:max_length]

    order = 5
    fs = x.shape[0]
    cutoff = determine_cutoff(x)
    if cutoff < fs / 10:
        cutoff = 5 * cutoff
    elif cutoff < fs / 4:
        cutoff = 2 * cutoff
    elif cutoff < fs / 2.2:
        cutoff = 1.1 * cutoff

    filtered_signal = lowpass_filter(x, cutoff, fs, order)
    noisy_signal = x - filtered_signal
    snr = calculate_snr(x, noisy_signal)
    return snr
