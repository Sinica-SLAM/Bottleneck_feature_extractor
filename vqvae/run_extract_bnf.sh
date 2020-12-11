#!/usr/bin/env bash

stage=3
stage_end=3
gpu=0
bnf_name=csid # "id" or "csid" or "token"
model_dir=exp/vqvae_librispeech
output_txt=true

[ -f path.sh ] && . ./path.sh; 
. utils/parse_options.sh || exit 1;

if [ -f vqvae/path.sh ]; then
    echo "Change the environment with vqvae/path.sh"
    . vqvae/path.sh
fi
. parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 <data-dir> <bnf-data-dir>"
  echo "e.g.: $0 data/train exp/train_bnf"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --gpu <gpu>                                      # ID of GPU."
  echo "  --bnf-name <csid>                                # bottlecneck feature type:"
  echo "                                                   # 1. token: VQ token"
  echo "                                                   # 2. id: VQ token ID"
  echo "                                                   # 3. csid: combined VQ token ID"
  echo "  --model-dir <exp/vqvae_librispeech>              # vqvae model dir."
  echo "  --output-txt <true>                              # output txt or not. if not, output ark and scp files"
  exit 1;
fi

data_dir=$1
bnf_data_dir=$2

# you might not want to do this for interactive shells.
set -e

if [ $stage -le 0 -a $stage_end -ge 0 ]; then
    data_name=`basename $data_dir`
    mkdir -p $feat_dir
    utils/copy_data_dir.sh $data_dir $bnf_data_dir/${data_name}_mel
    utils/data/resample_data_dir.sh 16000 $bnf_data_dir/${data_name}_mel
fi

# Feature preprocessing
if [ $stage -le 1 -a $stage_end -ge 1 ]; then
    # Extract features
    python vqvae/feature_extraction.py -c $model_dir/config_feature.json \
        -T $bnf_data_dir/${data_name}_mel -F $bnf_data_dir/${data_name}_mel/features \
        -K "mel"        

    mkdir -p $bnf_data_dir/${data_name}_mel_cmvn
    utils/copy_data_dir.sh $bnf_data_dir/${data_name}_mel $bnf_data_dir/${data_name}_mel_cmvn
    python vqvae/feature_normalization.py -c $model_dir/config_feature.json \
        -T $bnf_data_dir/${data_name}_mel_cmvn -F $bnf_data_dir/${data_name}_mel_cmvn/features \
        -S ${model_dir}/stats.pt \
        -K "mel"            
fi

# Extracing token feature
if [ $stage -le 2 -a $stage_end -ge 2 ]; then
    python vqvae/inference_bnf.py \
        -c $model_dir/config_vqvae.json \
        -d $bnf_data_dir/${data_name}_mel_cmvn \
        -o $bnf_data_dir \
        -K "mel-${bnf_name}" \
        -m ${model_dir}/model.pt -g $gpu \
        --output_txt $output_txt
fi

echo "Finish extracting VQVAE bottleneck features"
