#! python
# @Time    : 17-8-23
# @Author  : kay
# @File    : features.py
# @E-mail  : 861186267@qq.com
# @Function:

import numpy as np


def hz_to_mel(hz):
    """
    Convert Hz to Mel.
    :param hz: value in Hz
    :return: value in Mel
    """
    return 2595 * np.log10(1 + hz * 1.0 / 700)


def mel_to_hz(mel):
    """
    Convert Mel to Hz.
    :param mel: value in Mel
    :return: value in Hz
    """
    return 700 * (10 ** (mel * 1.0 / 2595) - 1)


def get_filterbanks(rate, filters_count=26, fft_size=512, low_freq=0, high_freq=None):
    """
    Create Mel-filterbanks.
    :param rate: frame rate of audio signal
    :param filters_count: number of filters
    :param fft_size: n-points of discrete fourier transform
    :param low_freq: start frequency of first filter
    :param high_freq: end frequency of last filter
    :return: numpy array with filterbanks
    """
    if high_freq is None:
        high_freq = rate / 2

    assert high_freq <= rate / 2, "high_freq must be lower than rate / 2"

    # convert Hz to Mel
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)

    # calculate filter points linearly spaced between lowest and highest frequency
    mel_points = np.linspace(low_mel, high_mel, filters_count + 2)

    # convert points back to Hz
    hz_points = mel_to_hz(mel_points)

    # round frequencies to nearest fft bin
    fft_bin = np.floor((fft_size + 1) * hz_points / rate)

    # first filterbank will start at the first point, reach its peak at the second point
    # then return to zero at the 3rd point. The second filterbank will start at the 2nd
    # point, reach its max at the 3rd, then be zero at the 4th etc.
    filterbanks = np.zeros([int(filters_count), int(fft_size / 2 + 1)])

    for i in range(filters_count):
        # from left to peak
        for j in range(int(fft_bin[i]), int(fft_bin[i + 1])):
            filterbanks[i, j] = (j - fft_bin[i]) / (fft_bin[i + 1] - fft_bin[i])
        # from peak to right
        for j in range(int(fft_bin[i + 1]), int(fft_bin[i + 2])):
            filterbanks[i, j] = (fft_bin[i + 2] - j) / (fft_bin[i + 2] - fft_bin[i + 1])

    return filterbanks


def get_power_spectrum(frames, fft_size):
    """
    Calculate periodogram estimate of power spectrum (for each frame).
    :param frames:  framed audio signal
    :param fft_size: n-points of discrete fourier transform
    :return: power spectrum for each frame
    """
    # np.square is element-wise
    return 1.0 / fft_size * np.square(np.abs(np.fft.rfft(frames, fft_size)))


def get_frames(signal, rate, frame_length, frame_step, window_function=lambda x: np.ones((x,))):
    """
    Split signal into frames (with our without overlay) and apply window function.
    :param signal: audio signal
    :param rate: frame rate of audio signal
    :param frame_length: length of single frame (in seconds)
    :param frame_step: length of frame step (in seconds)
    :param window_function: window function to be applied to every frame (default = rectangular)
    :return: numpy array with single frames
    """
    signal_length = len(signal)

    # convert frame length and step size to samples per frame
    frame_length = int(rate * frame_length)
    frame_step = int(rate * frame_step)

    # calculate frame count
    if signal_length < frame_length:
        frames_count = 1
    else:
        frames_count = 1 + int(np.ceil((signal_length * 1.0 - frame_length) / frame_step))

    # if last frame is incomplete, add padding of zeroes
    padding_length = int((frames_count - 1) * frame_step + frame_length)
    padding = np.zeros((padding_length - signal_length))
    signal = np.concatenate((signal, padding))

    # create array with frame indexes
    indexes = np.tile(np.arange(0, frame_length), (frames_count, 1)) + np.tile(
        np.arange(0, frames_count * frame_step, frame_step), (frame_length, 1)).T

    # use indexes mask to get single frames
    frames = signal[indexes]

    # create window function mask
    windows = np.tile(window_function(frame_length), (frames_count, 1))

    return frames * windows


def get_filterbank_energies(signal, rate, frame_length, frame_step, window_function=lambda x: np.ones((x,)),
                            filters_count=26, fft_size=512, low_freq=0, high_freq=None, log=False):
    """
    Compute Mel-filterbank energies.
    :param signal: audio signal
    :param rate: frame rate of audio signal
    :param frame_length: length of single frame (in seconds)
    :param frame_step: length of frame step (in seconds)
    :param window_function: window function to be applied to every frame (default = rectangular)
    :param filters_count: number of filters
    :param fft_size: n-points of discrete fourier transform
    :param low_freq: start frequency of first filter
    :param high_freq: end frequency of last filter
    :param log: if True, compute log-filterbank energies
    :return: numpy array with (log-)filterbank energies
    """
    if high_freq is None:
        high_freq = rate / 2

    frames = get_frames(signal, rate, frame_length, frame_step, window_function)

    power_spectrum = get_power_spectrum(frames, fft_size)
    filterbanks = get_filterbanks(rate, filters_count, fft_size, low_freq, high_freq)

    # weighted sum of the fft energies around filterbank frequencies
    filterbank_energies = np.dot(power_spectrum, filterbanks.T)

    if log == True:
        # replace zeroes with machine epsilon to prevent errors in log operation
        filterbank_energies = np.where(filterbank_energies == 0, np.finfo(float).eps, filterbank_energies)
        return np.log(filterbank_energies)
    else:
        return filterbank_energies


def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


def scale(X, axis=0, with_mean=True, with_std=True):
    X = np.asarray(X)
    if with_mean:
        mean_ = np.mean(X, axis)
    if with_std:
        scale_ = np.std(X, axis)
    # Xr is a view on the original array that enables easy use of
    # broadcasting on the axis in which we are interested in
    Xr = np.rollaxis(X, axis)
    if with_mean:
        Xr -= mean_
        mean_1 = Xr.mean(axis=0)
        # Verify that mean_1 is 'close to zero'. If X contains very
        # large values, mean_1 can also be very large, due to a lack of
        # precision of mean_. In this case, a pre-scaling of the
        # concerned feature is efficient, for instance by its mean or
        # maximum.
        if not np.allclose(mean_1, 0):
            Xr -= mean_1
    if with_std:
        scale_ = _handle_zeros_in_scale(scale_, copy=False)
        Xr /= scale_
        if with_mean:
            mean_2 = Xr.mean(axis=0)
            # If mean_2 is not 'close to zero', it comes from the fact that
            # scale_ is very small so that mean_2 = mean_1/scale_ > 0, even
            # if mean_1 was close to zero. The problem is thus essentially
            # due to the lack of precision of mean_. A solution is then to
            # subtract the mean again:
            if not np.allclose(mean_2, 0):
                Xr -= mean_2
    return X
