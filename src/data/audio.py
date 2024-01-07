import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import sys
from src.data.hparams import hparams as hp

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size

def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def melspectrogram(wav, fmin=55):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D), fmin)) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def _lws_processor():
    import lws
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")

def _stft(y):
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size)

def num_frames(length, fsize, fshift):
    """ Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """ Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
    
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

_mel_basis = None

def _linear_to_mel(spectogram, fmin):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(fmin)
    return np.dot(_mel_basis, spectogram)

def _build_mel_basis(fmin):
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,
                               fmin=fmin, fmax=hp.fmax)

def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)
    
    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))