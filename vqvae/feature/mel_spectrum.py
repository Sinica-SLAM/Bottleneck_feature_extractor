import torch

from librosa.filters import mel as librosa_mel_fn



class MelSpectrum(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0, feat_stat=None):
        super(MelSpectrum, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

        window = torch.hann_window(win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()

        self.register_buffer('window', window)
        self.register_buffer('mel_basis', mel_basis)

        if feat_stat is not None:
            assert 'mel_mean' in feat_stat.keys()
            assert 'mel_std' in feat_stat.keys()
            mel_mean = feat_stat['mel_mean'].view(1,-1,1)
            mel_std = feat_stat['mel_std'].view(1,-1,1)
            assert mel_mean.size(1) == n_mel_channels
            assert mel_std.size(1) == n_mel_channels
            self.register_buffer('mel_mean', mel_mean)
            self.register_buffer('mel_std', mel_std)
            self.feat_stat = True
        else:
            self.feat_stat = False

    def forward(self, audio):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        audio: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(audio.data) >= -1)
        assert(torch.max(audio.data) <= 1)

        magnitudes = self.stft(audio)
        mel_output = torch.matmul(self.mel_basis, magnitudes)

        return {'mel':mel_output}

    def stft(self, x):
        """Perform STFT and convert to magnitude spectrogram.
        Args:
            x (Tensor): Input signal tensor (B, T).
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            win_length (int): Window length.
            window (str): Window function type.
        Returns:
            Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
        """
        x_stft = torch.stft(x, self.filter_length, self.hop_length, self.win_length, self.window)
        real = x_stft[..., 0]
        imag = x_stft[..., 1]

        # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
        return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-10))


    def statistic(self, feat, min_value=1e-10):
        stat_dict = dict()
        if 'mel' in feat.keys():
            mel = feat['mel']
            assert mel.ndim == 2
            mel = torch.log(torch.clamp(mel, min=min_value))
            mel_mean = torch.mean( mel, dim=-1).view(1,-1,1).float()
            mel_std = torch.std( mel, dim=-1).view(1,-1,1).float()
            self.register_buffer('mel_mean', mel_mean)
            self.register_buffer('mel_std', mel_std)
            stat_dict['mel_mean'] = mel_mean
            stat_dict['mel_std'] = mel_std
        return stat_dict

    def normalize(self, feat, min_value=1e-10):
        if 'mel' in feat.keys():            
            mel = feat['mel']
            mel = torch.log(torch.clamp(mel, min=min_value))
            if self.feat_stat:
                mel = (mel - self.mel_mean) / self.mel_std
            feat['mel'] = mel
        return feat

    def denormalize(self, feat):
        if 'mel' in feat.keys():
            mel = feat['mel']
            if self.feat_stat:
                mel = mel * self.mel_std + self.mel_mean
            feat['mel'] = mel.exp()
        return feat