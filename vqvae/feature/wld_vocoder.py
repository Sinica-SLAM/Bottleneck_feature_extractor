import torch
import numpy as np

from scipy.signal import firwin, lfilter
import pyworld
import pysptk
# Our tools
from .vad import extfrm

MAX_WAV_VALUE = 32768.0

class Wld_vocoder(torch.nn.Module):
    def __init__(self, fft_size=1024, shiftms=5, sampling_rate=22050, 
                 mcc_dim=24, mcc_alpha=0.41, minf0=40, maxf0=700,
                 cutoff_freq=70, feat_stat=None):
        super(Wld_vocoder, self).__init__()

        self.fft_size = fft_size
        self.shiftms = shiftms
        self.fs = sampling_rate
        self.mcc_dim = mcc_dim
        self.mcc_alpha = mcc_alpha
        self.minf0 = minf0
        self.maxf0 = maxf0

        self.cutoff_freq = cutoff_freq

        self.sp_dim = fft_size//2 + 1

        self.sp_stat = False
        self.mcc_stat = False
        self.f0_stat = False
        self.en_stat = False

        if feat_stat is not None:
            stat_list = feat_stat.keys()
            if 'sp_min' in stat_list and 'sp_scale' in stat_list:
                sp_min = feat_stat['sp_min'].view(1,-1,1)
                sp_scale = feat_stat['sp_scale'].view(1,-1,1)
                assert sp_min.size(1) == fft_size // 2 + 1
                assert sp_scale.size(1) == fft_size // 2 + 1
                self.register_buffer('sp_min', sp_min)
                self.register_buffer('sp_scale', sp_scale)
                self.sp_stat = True

            if 'mcc_min' in stat_list and 'mcc_scale' in stat_list:
                mcc_min = feat_stat['mcc_min'].view(1,-1,1)
                mcc_scale = feat_stat['mcc_scale'].view(1,-1,1)
                assert mcc_min.size(1) == mcc_dim
                assert mcc_scale.size(1) == mcc_dim
                self.register_buffer('mcc_min', mcc_min)
                self.register_buffer('mcc_scale', mcc_scale)
                self.mcc_stat = True

            if 'f0_mean' in stat_list and 'f0_std' in stat_list:
                f0_mean = feat_stat['f0_mean'].view(1,-1)
                f0_std = feat_stat['f0_std'].view(1,-1)
                assert f0_mean.size(1) == 1
                assert f0_std.size(1) == 1
                self.register_buffer('f0_mean', f0_mean)
                self.register_buffer('f0_std', f0_std)
                self.f0_stat = True

            if 'en_max' in stat_list and 'en_min' in stat_list:
                en_max = feat_stat['en_max'].view(1,-1)
                en_min = feat_stat['en_min'].view(1,-1)
                assert en_max.size(1) == 1
                assert en_min.size(1) == 1
                self.register_buffer('en_max', en_max)
                self.register_buffer('en_min', en_min)
                self.en_stat = True                


    def forward(self, audio, feat_kinds=['sp','mcc','f0','ap','en']):
        """Computes world features from a batch of waves
        PARAMS
        ------
        audio: Variable(torch.FloatTensor) with shape (T) in range [-1, 1]

        RETURNS
        -------
        feat: torch.FloatTensor of shape ((SP+MCC+F0+AP+1+1), T)
                Contains features in "feat_kinds": SP, MCC, F0, AP, SP_en, MCC_en
        """
        device = audio.device
        audio = audio.detach().cpu().numpy()
        feat = dict()
        for feat_kind in feat_kinds:
            feat[feat_kind] = list()

        for x in audio:
            # Preprocess
            x = x * MAX_WAV_VALUE
            x = self.low_cut_filter(x, cutoff=self.cutoff_freq)
            # Extract f0
            f0, time_axis = pyworld.harvest(x, self.fs, f0_floor=self.minf0, f0_ceil=self.maxf0, frame_period=self.shiftms)

            # Extract sp    
            sp = pyworld.cheaptrick(x, f0, time_axis, self.fs, fft_size=self.fft_size)
            if 'sp' in feat_kinds:
                feat['sp'].append(torch.from_numpy(sp).float().t())

            # Extract ap
            if 'ap' in feat_kinds:
                ap = pyworld.d4c(x, f0, time_axis, self.fs, fft_size=self.fft_size)
                feat['ap'].append(torch.from_numpy(ap).float().t())

            # Extract mcc
            if 'mcc' in feat_kinds:
                mcc = pysptk.sp2mc(sp, self.mcc_dim, self.mcc_alpha)
                feat['mcc'].append(torch.from_numpy(mcc).float().t())

            # Extract energy
            if 'en' in feat_kinds:
                mcc = pysptk.sp2mc(sp, self.mcc_dim, self.mcc_alpha)
                en = pysptk.mc2e(mcc, alpha=self.mcc_alpha, irlen=256)
                # en = np.clip(en, 1e-10, None)
                feat['en'].append(torch.from_numpy(en).float().view(-1))                

            # Fix f0
            if 'f0' in feat_kinds:
                f0[f0 < 0] = 0
                feat['f0'].append(torch.from_numpy(f0).float().view(-1))

        for key, val_list in feat.items():
            feat[key] = torch.cat([val.unsqueeze(0) for val in val_list],dim=0).to(device)

        return feat

    def synthesis(self, feat, se_kind='sp'):
        batch_size = feat['ap'].size(0)
        device = feat['ap'].device

        audio = []
        for i in range(batch_size):
            ap = feat['ap'][i].detach().t().cpu().double().numpy()
            f0 = feat['f0'][i].detach().view(-1).cpu().double().numpy()
            if se_kind == 'mcc':
                mcc = feat['mcc'][i].detach().t().cpu().double().numpy()
                sp = pysptk.mc2sp(mcc.copy(order='C'), self.mcc_alpha, self.fft_size)
            else:
                sp = feat['sp'][i].detach().t().cpu().double().numpy()

            syn = pyworld.synthesize(
                    f0.copy(order='C'), 
                    sp.copy(order='C'), 
                    ap.copy(order='C'), 
                    self.fs, 
                    frame_period=self.shiftms
                )
            audio.append(torch.from_numpy(syn).float().view(-1))

        audio = torch.cat([syn.unsqueeze(0) for syn in audio],dim=0).to(device)

        return audio / MAX_WAV_VALUE


    def low_cut_filter(self, x, cutoff=70):
        """Low cut filter

        Parameters
        ---------
        x : array, shape(`samples`)
            Waveform sequence
        cutoff : float, optional
            Cutoff frequency of low cut filter
            Default set to 70 [Hz]

        Returns
        ---------
        lcf_x : array, shape(`samples`)
            Low cut filtered waveform sequence
        """

        nyquist = self.fs // 2
        norm_cutoff = cutoff / nyquist

        # low cut filter
        fil = firwin(255, norm_cutoff, pass_zero=False)
        lcf_x = lfilter(fil, 1, x)

        return lcf_x


    def statistic(self, feat, min_value=1e-10):
        feat_kinds = feat.keys()
        stat_dict = dict()

        if 'sp' in feat_kinds:
            sp = feat['sp']
            assert sp.ndim == 2
            sp, _ = self.sp_en_norm(sp.t(),dim=1)
            sp = sp.clamp(min=min_value).log().detach().cpu().numpy()
            sp_max = np.percentile(sp, 99.5, axis=0)
            sp_min = np.percentile(sp, 0.5, axis=0)
            sp_max = torch.from_numpy(sp_max).view(1,-1,1).float()
            sp_min = torch.from_numpy(sp_min).view(1,-1,1).float()
            sp_scale = sp_max - sp_min
            self.register_buffer('sp_min', sp_min)
            self.register_buffer('sp_scale', sp_scale)
            self.sp_stat = True
            stat_dict['sp_min'] = sp_min
            stat_dict['sp_scale'] = sp_scale
        
        if 'mcc' in feat_kinds:
            mcc = feat['mcc']
            assert mcc.ndim == 2
            mcc = mcc[1:].t().detach().cpu().numpy()
            mcc_max = np.percentile(mcc, 99.5, axis=0)
            mcc_min = np.percentile(mcc, 0.5, axis=0)
            mcc_var = np.var(mcc, axis=0)
            mcc_max = torch.from_numpy(mcc_max).view(1,-1,1).float()
            mcc_min = torch.from_numpy(mcc_min).view(1,-1,1).float()
            mcc_var = torch.from_numpy(mcc_var).view(1,-1,1).float()
            mcc_scale = mcc_max - mcc_min
            self.register_buffer('mcc_min', mcc_min)
            self.register_buffer('mcc_scale', mcc_scale)
            self.mcc_stat = True
            stat_dict['mcc_min'] = mcc_min
            stat_dict['mcc_scale'] = mcc_scale
            stat_dict['mcc_var'] = mcc_var

        if 'f0' in feat_kinds:
            f0 = feat['f0']
            assert f0.ndim == 1
            f0 = f0[f0 > 0].log2()
            f0_mean = torch.mean(f0).float()
            f0_std = torch.std(f0).float()
            self.register_buffer('f0_mean', f0_mean)
            self.register_buffer('f0_std', f0_std)
            self.f0_stat = True
            stat_dict['f0_mean'] = f0_mean
            stat_dict['f0_std'] = f0_std

        if 'en' in feat_kinds:
            en = feat['en']
            assert en.ndim == 1
            en = en.clamp(min=min_value).log().detach().cpu().numpy()
            en_max = np.percentile(en, 99.5, axis=0)
            en_min = np.percentile(en, 0.5, axis=0)
            self.register_buffer('en_max', en_max)
            self.register_buffer('en_min', en_min)
            self.en_stat = True
            stat_dict['en_max'] = en_max
            stat_dict['en_min'] = en_min              

        return stat_dict

    def normalize(self, feat, min_value=1e-10):
        feat_kinds = feat.keys()

        if 'sp' in feat_kinds:
            sp = feat['sp']
            sp, sp_en = self.sp_en_norm(sp,dim=1)
            sp = sp.clamp(min=min_value).log()
            if self.sp_stat:
                sp = (sp - self.sp_min) / self.sp_scale
                sp = sp.clamp(min=0.0,max=1.0) * 2 - 1
            feat['sp'] = sp
            feat['sp_en'] = sp_en

        if 'mcc' in feat_kinds:
            mcc = feat['mcc'][:,1:]
            mcc_en = feat['mcc'][:,:1]
            if self.mcc_stat:
                mcc = (mcc - self.mcc_min) / self.mcc_scale
                mcc = mcc.clamp(min=0.0,max=1.0) * 2 - 1
            feat['mcc'] = mcc
            feat['mcc_en'] = mcc_en

        if 'f0' in feat_kinds:
            f0 = feat['f0']
            f0[f0 > 0] = f0[f0 > 0].log2()
            if self.f0_stat:
                f0[f0 > 0] = (f0[f0 > 0] - self.f0_mean) / self.f0_std
            feat['f0'] = f0

        if 'en' in feat_kinds:
            en = feat['en']
            en = en.clamp(min=min_value).log()
            if self.en_stat:
                en = (en - self.en_min) / (self.en_max - en_min)
            feat['en'] = en            

        return feat

    def denormalize(self, feat):
        feat_kinds = feat.keys()
        if 'sp' in feat_kinds:
            sp = feat['sp']
            if self.sp_stat:
                sp = (sp * 0.5 + 0.5) * self.sp_scale + self.sp_min
            sp = sp.exp()
            feat['sp'] = self.sp_en_denorm(sp, feat['sp_en'])

        if 'mcc' in feat_kinds:
            mcc = feat['mcc']
            if self.mcc_stat:
                mcc = (mcc * 0.5 + 0.5) * self.mcc_scale + self.mcc_min
            feat['mcc'] = torch.cat([feat['mcc_en'],mcc],dim=1)

        if 'f0' in feat_kinds:
            f0 = feat['f0']
            if self.f0_stat:
                f0[f0 > 0] = f0[f0 > 0] * self.f0_std + self.f0_mean
            f0[f0 > 0] = 2**(f0[f0 > 0])
            feat['f0'] = f0

        return feat

    def sp_en_norm(self, sp, dim=1, eps=1e-10):
        en = torch.sum( sp + eps, dim=dim, keepdims=True)
        sp = sp / en
        return sp, en

    def sp_en_denorm(self, sp, en):
        sp = en * sp
        return sp
