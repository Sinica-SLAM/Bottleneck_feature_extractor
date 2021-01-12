#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path
from importlib import import_module

import numpy as np
import librosa
import math

import torch
import torch.nn.functional as F

from kaldiio import ReadHelper, WriteHelper

from feature.mel_spectrum import MelSpectrum

MAX_WAV_VALUE = 32768.0


def inference( 
        rspecifier, wspecifier, 
        model_type, model_path, model_config, 
        bnf_feature_kind, data_config, 
        output_txt
    ):

    stat_dict       = data_config.get('statistic_file', None)
    feat_kind       = data_config.get('feature_kind', 'mel-id')
    filter_length   = data_config.get('filter_length', 1024)
    hop_length      = data_config.get('hop_length', 256)
    win_length      = data_config.get('win_length', 1024)
    n_mel_channels  = data_config.get('n_mel_channels', 80)
    sampling_rate   = data_config.get('sampling_rate', 24000)
    mel_fmin        = data_config.get('mel_fmin', 80)
    mel_fmax        = data_config.get('mel_fmax', 7600)   

    feature_kinds = feat_kind.split('-')
    assert feature_kinds[0] in ['mel']
    assert bnf_feature_kind in ['id','csid','token']

    module = import_module('model.{}'.format(model_type), package=None)
    model = getattr(module, 'Model')( model_config)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    model.cuda().eval()

    # Read stat scp
    if stat_dict is None:
        with open(data_config.get('training_dir','') / 'stat.scp', 'r') as rf:
            stat_dict = [line.rstrip() for line in rf.readlines()][0]
    elif stat_dict.split('.')[-1] == 'scp':
        with open(stat_dict, 'r') as rf:
            stat_dict = [line.rstrip() for line in rf.readlines()][0]

    feat_stat = torch.load(stat_dict)

    feat_fn = MelSpectrum(  
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        sampling_rate=sampling_rate,
        mel_fmin=mel_fmin, mel_fmax=mel_fmax,
        feat_stat=feat_stat
    ).cuda()

    if output_txt and bnf_feature_kind in ['id','csid']:
        bnf_writer = open(wspecifier,'w')
    else:
        bnf_writer = WriteHelper(bnf_writer, compression_method=1)
        output_txt = False

    for utt, (rate, X) in ReadHelper(rspecifier):
        X = X.astype(np.float32) / MAX_WAV_VALUE
        X = librosa.core.resample(
                X, 
                rate, 
                sampling_rate, 
                res_type='kaiser_best'
            )
        if np.max(np.abs(X)) >= 1.0:
            X /= np.max(np.abs(X))
        # Extract features
        X = feat_fn(torch.from_numpy(X).cuda().unsqueeze(0))
        X = feat_fn.normalize(X)

        X_in = X['mel']
   
        with torch.no_grad():
            z = model.encoder(X_in)
            z_id = model.quantizer.encode(z)
            z_vq = model.quantizer.decode(z_id)
            
        # Save converted feats
        if bnf_feature_kind == 'id':
            X_bnf = z_id.view(-1).cpu().numpy()
        if bnf_feature_kind == 'csid':
            X_bnf = z_id.view(-1).unique_consecutive().cpu().numpy()
        elif bnf_feature_kind == 'token':
            X_bnf = z_vq.squeeze(0).t().cpu().numpy()

        if output_txt:
            X_bnf = X_bnf.reshape(-1)
            X_bnf = ''.join(['<{}>'.format(bnf) for bnf in X_bnf])
            bnf_writer.write('{} {}\n'.format(utt,X_bnf))
        else:
            bnf_writer.write(utt, X_bnf)
        
        print('Extracting BNF {} of {}.'.format( bnf_feature_kind, utt),end=' '*30+'\r')

    bnf_writer.close()


if __name__ == "__main__":
    import argparse
    import json
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/config_vae_vctk.json',
                        help='JSON file for configuration')
    parser.add_argument('-m','--model_path', type=str, default=None,
                        help='Path to checkpoint with model')
    parser.add_argument('-s','--statistic_file', type=str, default=None,
                        help='Statistic file path')
    parser.add_argument('-b','--bnf_feature_kind', type=str, default='id',
                        help='Feature kinds')
    parser.add_argument('-t','--output_txt', type=str, default='true')
    parser.add_argument('-g', "--gpu", type=str, default='0')
    parser.add_argument('rspecifier', type=str,
                        help='Input specifier')
    parser.add_argument('wspecifier', type=str,
                        help='Output specifier')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load Config.
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    infer_config = config["infer_config"] 
    data_config = config["data_config"]

    if args.model_path is None:
        model_path = infer_config["model_path"]
    else:
        model_path = args.model_path

    model_type = infer_config["model_type"]    

    # Fix model config if GAN is used
    if 'Generator' in config["model_config"].keys():
        model_config = config["model_config"]['Generator']
    else:
        model_config = config["model_config"]

    if args.statistic_file is not None:
        data_config['statistic_file'] = args.statistic_file

    output_txt = True if args.output_txt.lower() in ['true'] else False

    inference( 
        args.rspecifier, args.wspecifier,
        model_type, model_path, model_config, 
        args.bnf_feature_kind, data_config, 
        output_txt
    )
