import os
import json
import numpy as np
from pathlib import Path

import torch
import random
from scipy.io import wavfile
import soundfile

import librosa

from kaldi.util.table import SequentialWaveReader

from feature.stft import STFT
from feature.mel_spectrum import MelSpectrum
from feature.wld_vocoder import Wld_vocoder

MAX_WAV_VALUE = 32768.0
MAX_MAT_NUM = 30000
MIN_SPEC_VALUE = 1e-10


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(str(filename), 'r') as f:
        files = f.readlines()

    files = [f.rstrip().split() for f in files]
    return files

def load_wav_to_torch(full_path, target_sampling_rate, trim_silence=False):
    """
    Loads wavdata into torch array
    """

    if isinstance(full_path, str):
        ext = os.path.splitext(full_path)[-1].lower()
        if ext in ['.wav']:
            data_sampling_rate, data = wavfile.read(full_path)
            data = data.astype(np.float32) / MAX_WAV_VALUE
        elif ext in ['.flac']:
            data, data_sampling_rate = soundfile.read(full_path)
        else:
            raise ValueError("File format {}... not understood.".format(ext))

        if data_sampling_rate != target_sampling_rate:
            data = librosa.core.resample(
                    data, 
                    data_sampling_rate, 
                    target_sampling_rate, 
                    res_type='kaiser_best'
                )
            # raise ValueError("{} SR doesn't match target {} SR".format(
            #     data_sampling_rate, target_sampling_rate))            
    else:
        data = full_path.data().numpy().astype(np.float32) / MAX_WAV_VALUE
    
    if trim_silence:
        data,_ = librosa.effectstrim(  
                data,
                top_db=25,
                frame_length=1024,
                hop_length=256
            )

    if np.max(np.abs(data)) >= 1.0:
        data /= np.max(np.abs(data))

    return torch.from_numpy(data).float()


def main(config):
    training_dir    = config.get('training_dir', '')
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
    trim_silence    = config.get('trim_silence', False)

    training_dir = Path(training_dir)
    assert feature_dir is not None and feature_dir not in ['none']
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    feature_kinds = feature_kind.split('-')
    print('Extract: {}'.format(feature_kinds))
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
        wld_fn = Wld_vocoder(   fft_size=fft_size, 
                                shiftms=shiftms, 
                                sampling_rate=sampling_rate, 
                                mcc_dim=mcc_dim, mcc_alpha=mcc_alpha, 
                                minf0=f0_fmin, maxf0=f0_fmax,
                                cutoff_freq=cutoff_freq)
        func_kinds.append(['wld',wld_fn])

    feat_scp = open( training_dir / 'feats.scp', 'w')

    # Use normal wav list or kaldi tool
    data_list = files_to_list( training_dir / 'wav.scp')
    if sum([1 for d in data_list if len(d) > 2]) > 0:
        scp_file = 'scp:{}'.format(str(training_dir / 'wav.scp'))
        data_list = SequentialWaveReader(scp_file)


    for data_name, data in data_list:
        print('Data : {}'.format(data_name),end=' '*30+'\r')

        audio = load_wav_to_torch(data, sampling_rate, trim_silence=trim_silence)

        feat = dict()
        if 'wav' in feature_kinds:
            feat['wav'] = audio.squeeze(0).float()
            
        for func_kind, feat_fn in func_kinds:
            _feat = feat_fn(audio)

            for key,val in _feat.items():
                if key in feature_kinds:
                    feat[key] = val.squeeze(0).float()



        output_path = feature_dir / '{}.pt'.format(data_name)
        feat_scp.write('{} {}\n'.format(data_name,str(output_path.absolute())))
        torch.save( feat, output_path)

    feat_scp.close()


if __name__ == '__main__':
 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/config_feature.json',
                        help='JSON file for configuration')
    parser.add_argument('-T', '--training_dir', type=str, default=None,
                        help='Traininig dictionary path')
    parser.add_argument('-F', '--feature_dir', type=str, default=None,
                        help='Feature data dir')
    parser.add_argument('-K', '--feature_kind', type=str, default=None,
                        help='Feature kind')
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

    main(data_config)

