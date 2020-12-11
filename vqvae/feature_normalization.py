import os
import json
import numpy as np
from pathlib import Path

import torch
import random

from feature.stft import STFT
from feature.mel_spectrum import MelSpectrum
from feature.wld_vocoder import Wld_vocoder


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(str(filename), 'r') as f:
        files = f.readlines()

    files = [f.rstrip().split() for f in files]
    return files


def main(config):
    training_dir    = config.get('training_dir', '')
    statistic_file  = config.get('statistic_file', '')
    feature_kind    = config.get('feature_kind', 'mel')
    feature_dir     = config.get('feature_dir', 'none')
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
    assert feature_dir is not None and feature_dir not in ['none']
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    if statistic_file == '':
        with open(training_dir / 'stat.scp', 'r') as rf:
            statistic_file = [line.rstrip() for line in rf.readlines()][0]
    elif statistic_file.split('.')[-1] == 'scp':
        with open(statistic_file, 'r') as rf:
            statistic_file = [line.rstrip() for line in rf.readlines()][0]
            
    feat_stat = torch.load(statistic_file)
    feature_kinds = feature_kind.split('-')
    print('Normalize: {}'.format(feature_kinds))
    func_kinds = []
    stat_kinds = []

    if 'stft' in feature_kinds:
        stft_fn = STFT( filter_length=fft_size,
                        hop_length=hop_length, 
                        win_length=win_length, 
                        window='hann',
                        feat_stat=feat_stat)
        func_kinds.append(['stft',stft_fn])
        if stft_fn.feat_stat:
            stat_kinds.append('stft')

    if 'mel' in feature_kinds:
        mel_fn = MelSpectrum(  filter_length=fft_size,
                                hop_length=hop_length,
                                win_length=win_length,
                                n_mel_channels=n_mel_channels,
                                sampling_rate=sampling_rate,
                                mel_fmin=mel_fmin, mel_fmax=mel_fmax,
                                feat_stat=feat_stat)
        func_kinds.append(['mel',mel_fn])
        if mel_fn.feat_stat:
            stat_kinds.append('mel')
     
    if sum([(f in feature_kinds) for f in ['sp','mcc','f0','ap','wld']]) > 0:
        wld_fn = Wld_vocoder(  fft_size=fft_size, 
                                shiftms=shiftms, 
                                sampling_rate=sampling_rate, 
                                mcc_dim=mcc_dim, mcc_alpha=mcc_alpha, 
                                minf0=f0_fmin, maxf0=f0_fmax,
                                cutoff_freq=cutoff_freq,
                                feat_stat=feat_stat)
        func_kinds.append(['wld',wld_fn])
        if wld_fn.sp_stat:
            stat_kinds.append('sp')
        if wld_fn.mcc_stat:
            stat_kinds.append('mcc')
        if wld_fn.f0_stat:
            stat_kinds.append('f0')

    print('Load statistic: {}'.format(stat_kinds))

    data_list = files_to_list( training_dir / 'feats.scp')
    feat_scp = open( training_dir / 'feats.scp', 'w')

    for data_name, data_path in data_list:
        print('Data : {}'.format(data_name),end=' '*30+'\r')

        feat = torch.load(data_path)
        feat_norm = dict()
        for key, val in feat.items():
            if key in feature_kinds:
                feat_norm[key] = val.unsqueeze(0)

        for func_kind, feat_fn in func_kinds:
            feat_norm = feat_fn.normalize(feat_norm)

        for key, val in feat_norm.items():
            feat[key] = val.squeeze(0)          

        output_path = feature_dir / '{}.pt'.format(data_name)
        feat_scp.write('{} {}\n'.format(data_name,str(output_path.absolute())))
        torch.save( feat, output_path)

    feat_scp.close()


if __name__ == '__main__':
 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/config_vae_vctk.json',
                        help='JSON file for configuration')
    parser.add_argument('-T', '--training_dir', type=str, default=None,
                        help='Traininig dictionary path')
    parser.add_argument('-F', '--feature_dir', type=str, default=None,
                        help='Input feature data dir')
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
    if args.feature_dir is not None:
        data_config['feature_dir'] = args.feature_dir 
    if args.feature_kind is not None:
        data_config['feature_kind'] = args.feature_kind        
    if args.statistic_file is not None:
        data_config['statistic_file'] = args.statistic_file

    main(data_config)

