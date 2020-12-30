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

# from kaldi.util.table import VectorWriter, MatrixWriter
from kaldiio import ReadHelper

from feature.mel_spectrum import MelSpectrum
from model.layers_vq import VectorQuantizer

MAX_WAV_VALUE = 32768.0

class Model(torch.nn.Module):
    def __init__(self, model_type, arch):
        super(Model, self).__init__()

        module = import_module('model.{}'.format(model_type), package=None)

        self.encoder = getattr(module, 'Encoder')(**arch['encoder'])
        self.quantizer = VectorQuantizer( arch['z_num'], arch['z_dim'], normalize=arch['embed_norm'], reduction='sum')
        
        self.beta = arch['beta']
        self.y_num = arch['y_num']

    def forward(self, input):
        x = input  
        # Encode
        z = self.encoder(x)

        return z

    def get_codebook(self):
        return self.quantizer._embedding.data


    def load_state_dict(self, state_dict):
        warning_mseg =  'Embedding size mismatch for {}: '
        warning_mseg += 'copying a param with shape {} from checkpoint, '
        warning_mseg += 'resizing the param with shape {} in current model.'

        state_dict_shape, module_param_shape = state_dict['quantizer._embedding'].shape, self.quantizer._embedding.shape
        if state_dict_shape != module_param_shape:
            print(warning_mseg.format('model.quantizer', state_dict_shape, module_param_shape))
            self.quantizer = VectorQuantizer( 
                    state_dict_shape[0], state_dict_shape[1], 
                    normalize=self.quantizer.normalize, reduction=self.quantizer.reduction
                    )
        state_dict_new = dict()
        for key, param in state_dict.items():
            if key.split('.')[0] in ['encoder','quantizer']:
                state_dict_new[key] = param

        super(Model, self).load_state_dict(state_dict_new)


def inference( wav_scp, bnf_feature_kind, model_type, model_path, output_wf):

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

    model = Model( model_type, model_config)

    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    model.cuda().eval()

    codebook = model.get_codebook()
    codebook = codebook / codebook.norm(dim=1, keepdim=True)

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


    with ReadHelper('scp:{}'.format(wav_scp)) as reader:
        for utt, (rate, X) in reader:
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

            X_in = X[feature_kinds[0]]
       
            with torch.no_grad():
                z = model(X_in)
                z = z.squeeze(0).t()
                distances = (torch.sum(z.pow(2), dim=1, keepdim=True) 
                            + torch.sum(codebook.pow(2), dim=1)
                            - 2 * torch.matmul(z, codebook.t()))
                z_id = torch.argmin(distances, dim=1)
                z_vq = codebook.index_select(dim=0, index=z_id)
                z_vq = z_vq.t()
                
                counter = torch.zeros(z_id.size(0), codebook.size(0), device=z_id.device)
                counter.scatter_(1, z_id.unsqueeze(1), 1)
                count = counter.sum(dim=0)
                
            # Save converted feats
            if bnf_feature_kind == 'id':
                X_bnf = z_id.unsqueeze(-1).cpu().numpy()
            if bnf_feature_kind == 'csid':
                X_bnf = z_id.unique_consecutive().unsqueeze(-1).cpu().numpy()

            X_bnf = X_bnf.reshape(-1)
            X_bnf = ''.join(['<{}>'.format(X_) for X_ in X_bnf])
            output_wf.write('{} {}\n'.format(utt,X_bnf))
            
            print('Extracting BNF {} of {}.'.format( bnf_feature_kind, utt),end=' '*30+'\r')

    if output_wf is not sys.stdout:
        output_wf.close()


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
    parser.add_argument('-b','--bnf_feature_kind', type=str, default=None,
                        help='Feature kinds')
    parser.add_argument('-g', "--gpu", type=str, default='0,1')
    parser.add_argument('-w','--wav_scp', type=str, default=None,
                        help='Dir. for input data')    
    parser.add_argument('-o', "--output_text_file", type=str, default=None)  
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load Config.
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    infer_config = config["infer_config"] 
    global data_config
    data_config = config["data_config"]
    global model_config

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

    # Load wav.scp
    if args.wav_scp is None:
        wav_scp = Path(data_config["testing_dir"]) / 'wav.scp'
    else:
        wav_scp = Path(args.wav_scp)

    # Make dir. for outputing converted feature
    if args.output_text_file is None:
        output_wf = sys.stdout
    else:
        output_wf = open(args.output_text_file,'w')

    inference( wav_scp, args.bnf_feature_kind, model_type, model_path, output_wf)
