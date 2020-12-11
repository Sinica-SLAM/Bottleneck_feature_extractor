import os
import json
import numpy as np
from pathlib import Path

import torch
import random
from scipy.io.wavfile import read
from librosa.effects import  trim
import librosa

from feature.stft import STFT
from feature.mel_spectrum import MelSpectrum
from feature.wld_vocoder import Wld_vocoder

MAX_WAV_VALUE = 32768.0
MAX_MAT_NUM = 30000
MIN_SPEC_VALUE = 1e-10
TRIM_SILENCE = True

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, 'r') as f:
        files = f.readlines()

    files = [f.rstrip().split() for f in files]
    return files

def load_wav_to_torch(full_path, target_sampling_rate):
    """
    Loads wavdata into torch array
    """
    # data_sampling_rate, data = read(full_path)
    # data = data / MAX_WAV_VALUE
    data, data_sampling_rate = librosa.core.load(full_path, sr=target_sampling_rate)

    if data_sampling_rate != target_sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            data_sampling_rate, target_sampling_rate))
    
    if TRIM_SILENCE:
        data,_ = trim(  data,
                        top_db=25,
                        frame_length=1024,
                        hop_length=256) 
    return torch.from_numpy(data).float()


def statistic(config):
    training_dir    = config.get('training_dir', '')
    statistic_file  = config.get('statistic_file', '')
    feature_kind    = config.get('feature_kind', 'mel')
    fft_size        = config.get('fft_size', 1024)
    hop_length      = config.get('hop_length', 256)
    shiftms         = config.get('shiftms', 5)
    win_length      = config.get('win_length', 1024)
    n_mel_channels  = config.get('n_mel_channels', 80)
    mcc_dim         = config.get('mcc_dim', 24)
    mcc_alpha       = config.get('mcc_alpha', 0.41)
    sampling_rate   = config.get('sampling_rate', 24000)
    mel_fmin        = config.get('mel_fmin', 80)
    mel_fmax        = config.get('mel_fmax', 7600)
    f0_fmin         = config.get('f0_fmin', 40)
    f0_fmax         = config.get('f0_fmax', 700)
    cutoff_freq     = config.get('cutoff_freq', 70)

    training_dir = Path(training_dir)
    if statistic_file == '':
        statistic_file = training_dir / 'stats.pt'
    else:
        statistic_file = Path(statistic_file)

    feature_kinds = feature_kind.split('-')
    print('Statistic: {}'.format(feature_kinds))
    func_kinds = []

    if 'stft' in feature_kinds:
        stft_fn = STFT( filter_length=fft_size,
                        hop_length=hop_length, 
                        win_length=win_length, 
                        window='hann')
        func_kinds.append(['stft',stft_fn])

    if 'mel' in feature_kinds:
        mel_fn = MelSpectrum(  filter_length=fft_size,
                                hop_length=hop_length,
                                win_length=win_length,
                                n_mel_channels=n_mel_channels,
                                sampling_rate=sampling_rate,
                                mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        func_kinds.append(['mel',mel_fn])
     

    if sum([(f in feature_kinds) for f in ['sp','mcc','f0','ap','wld']]) > 0:
        wld_fn = Wld_vocoder(  fft_size=fft_size, 
                                shiftms=shiftms, 
                                sampling_rate=sampling_rate, 
                                mcc_dim=mcc_dim, mcc_alpha=mcc_alpha, 
                                minf0=f0_fmin, maxf0=f0_fmax,
                                cutoff_freq=cutoff_freq)
        func_kinds.append(['wld',wld_fn])     

    if not (training_dir / 'feats.scp').exists():
        data_list = files_to_list(training_dir / 'wav.scp')
        use_feat = False
    else:
        data_list = files_to_list(training_dir / 'feats.scp')
        use_feat = True

    feat_dict = dict()
    for key in feature_kinds:
        feat_dict[key] = list()

    for data_name, data_path in data_list:
        print('Data : {}'.format(data_name),end=' '*30+'\r')

        if use_feat:
            _feat = torch.load(data_path)
            for key in feature_kinds:
                feat_dict[key].append(_feat[key])
        else:
            audio = load_wav_to_torch(data_path, sampling_rate)
            audio = audio.unsqueeze(0)
            for func_kind, feat_fn in func_kinds:
                _feat = feat_fn(audio)
                for key in feature_kinds:
                    feat_dict[key].append(_feat[key])

    print('Collect {} data for ftStat.'.format( len(data_list)),end=' '*30+'\n')

    
    for key in feature_kinds:
        feat_dict[key] = torch.cat(feat_dict[key], dim=-1)

    feat_stat = dict()

    for func_kind, feat_fn in func_kinds:
        stat = feat_fn.statistic(feat_dict)

        for key, val in stat.items():
            feat_stat[key] = val

    with open(training_dir / 'stats.scp', 'w') as stat_scp:
        stat_scp.write('{}\n'.format(str(statistic_file.absolute())))

    print('Fianlly obtain statistic of: {}'.format(feat_stat.keys()))

    torch.save( feat_stat, statistic_file)


if __name__ == '__main__':
 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/config_vae_vctk.json',
                        help='JSON file for configuration')
    parser.add_argument('-T', '--training_dir', type=str, default=None,
                        help='Traininig dictionary path')
    parser.add_argument('-K', '--feature_kind', type=str, default=None,
                        help='Feature kind')   
    parser.add_argument('-S', '--statistic_file', type=str, default=None,
                        help='Statistic file path')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    data_config = config["data_config"]

    if args.training_dir is not None:
        data_config['training_dir'] = args.training_dir
    if args.feature_kind is not None:
        data_config['feature_kind'] = args.feature_kind        
    if args.statistic_file is not None:
        data_config['statistic_file'] = args.statistic_file

    statistic(data_config)

