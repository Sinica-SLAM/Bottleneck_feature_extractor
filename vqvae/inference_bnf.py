import os
import time
from pathlib import Path
from importlib import import_module

import numpy as np
import math

import torch
import torch.nn.functional as F

from kaldi.util.table import VectorWriter, MatrixWriter

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


def inference( utt2path, model_type, model_path, output_feat_dir, output_txt):

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

    
    if output_txt and feature_kinds[-1] in ['id','csid']:
        wspecifier = '{0}/raw_bnfeat_{1}2{2}.txt'
        feat_writer = open(wspecifier.format(str(output_feat_dir),feature_kinds[0],feature_kinds[-1]),'w')
    else:
        wspecifier = 'ark,scp:{0}/raw_bnfeat_{1}.ark,{0}/raw_bnfeat_{1}.scp'
        feat_writer = MatrixWriter(wspecifier.format(str(output_feat_dir),feature_kinds[0]))
        output_txt = False

    for i, (utt,path) in enumerate(utt2path):
        # Load source features
        X = torch.load(path,map_location='cpu')
        X_in = X[feature_kinds[0]].cuda().unsqueeze(0)
        X_mel = X['mel']
   
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
        if feature_kinds[-1] == 'id':
            X_bnf = z_id.unsqueeze(-1).cpu().numpy()
        if feature_kinds[-1] == 'csid':
            X_bnf = z_id.unique_consecutive().unsqueeze(-1).cpu().numpy()
        elif feature_kinds[-1] == 'token':
            X_bnf = z_vq.t().cpu().numpy()
        elif feature_kinds[-1] == 'count':
            X_bnf = count.unsqueeze(0).cpu().numpy()

        if output_txt:
            X_bnf = X_bnf.reshape(-1)
            X_bnf = ''.join(['<{}>'.format(X) for X in X_bnf])
            feat_writer.write('{} {}\n'.format(utt,X_bnf))
        else:
            feat_writer.write(utt, X_bnf)
        
        print('Extracting BNF {} of {}.'.format( feature_kinds[-1], utt),end=' '*30+'\r')

    feat_writer.close()


if __name__ == "__main__":
    import argparse
    import json
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/config_vae_vctk.json',
                        help='JSON file for configuration')
    parser.add_argument('-d','--data_dir', type=str, default=None,
                        help='Dir. for input data')
    parser.add_argument('-m','--model_path', type=str, default=None,
                        help='Path to checkpoint with model')                    
    parser.add_argument('-K','--feature_kind', type=str, default=None,
                        help='Feature kinds')

    parser.add_argument('-o', "--output_feat_dir", type=str, default=None)
    
    parser.add_argument('-g', "--gpu", type=str, default='0,1')

    parser.add_argument('-t', "--output_txt", type=str, default='false')

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

    if args.feature_kind is not None:
        data_config['feature_kind'] = args.feature_kind

    # Load wav.scp & spk2spk_id
    if args.data_dir is None:
        data_dir = Path(data_config["testing_dir"])
    else:
        data_dir = Path(args.data_dir)
    with open(data_dir / 'feats.scp','r') as rf:
        utt2path = [line.rstrip().split() for line in rf.readlines()]

    # Make dir. for outputing converted feature
    if args.output_feat_dir is None:
        output_feat_dir = infer_config["output_feat_dir"]
    else:
        output_feat_dir = args.output_feat_dir
    if output_feat_dir is not None:
        output_feat_dir = Path(output_feat_dir)
        output_feat_dir.mkdir(parents=True, exist_ok=True)
   
    output_txt = True if args.output_txt.lower() in ['true'] else False

    inference( utt2path, model_type, model_path, output_feat_dir, output_txt)
