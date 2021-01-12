import os
import time
from pathlib import Path
from importlib import import_module

import numpy as np
import math

import torch
import torch.nn.functional as F

# from kaldi.util.table import VectorWriter, MatrixWriter
from kaldiio import ReadHelper, WriteHelper

MAX_WAV_VALUE = 32768.0


def inference( utt2path, model_type, model_path, model_config, data_config, output_feat_dir, output_txt):

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

    module = import_module('model.{}'.format(model_type), package=None)
    model = getattr(module, 'Model')( model_config)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    model.cuda().eval()
    
    if output_txt and feature_kinds[-1] in ['id','csid']:
        wspecifier = '{0}/raw_bnfeat_{1}2{2}.txt'
        feat_writer = open(wspecifier.format(str(output_feat_dir),feature_kinds[0],feature_kinds[-1]),'w')
    else:
        wspecifier = 'ark,scp:{0}/raw_bnfeat_{1}.ark,{0}/raw_bnfeat_{1}.scp'
        feat_writer = WriteHelper(
            wspecifier.format(str(output_feat_dir),feature_kinds[0]), 
            compression_method=1)
        output_txt = False

    for i, (utt,path) in enumerate(utt2path):
        # Load source features
        X = torch.load(path,map_location='cpu')
        X_in = X[feature_kinds[0]].cuda().unsqueeze(0)
        X_mel = X['mel']
   
        with torch.no_grad():
            z = model.encoder(X_in)
            z_id = model.quantizer.encode(z)
            z_vq = model.quantizer.decode(z_id)
            
        # Save converted feats
        if feature_kinds[-1] == 'id':
            X_bnf = z_id.view(-1).cpu().numpy()
        if feature_kinds[-1] == 'csid':
            X_bnf = z_id.view(-1).unique_consecutive().cpu().numpy()
        elif feature_kinds[-1] == 'token':
            X_bnf = z_vq.squeeze(0).t().cpu().numpy()

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

    inference( 
        utt2path, 
        model_type, model_path, model_config, 
        data_config, 
        output_feat_dir, output_txt
    )
