#!/usr/bin/env bash

data_dir=

stage=3
stage_end=3
gpu=0
bnf_name=id # "id" or "csid" or "token"
model_dir=exp/vqvae_librispeech

clean_up=false

[ -f path.sh ] && . ./path.sh; 
. utils/parse_options.sh || exit 1;

# you might not want to do this for interactive shells.
set -e

exp_dir=exp
feat_dir=exp/feature

if [ $stage -le 0 -a $stage_end -ge 0 ]; then
    task=`basename $data_dir`
    mkdir -p $exp_dir/$task
    mkdir -p $feat_dir
    cp $data_dir/wav.scp $exp_dir/$task/
    utils/data/resample_data_dir.sh 16000 $exp_dir/$task
fi

# Feature preprocessing
if [ $stage -le 1 -a $stage_end -ge 1 ]; then
    # Extract features
    python feature_extraction.py -c $model_dir/config_feature.json \
        -T $exp_dir/$task -F $feat_dir/${task} \
        -K "mel"        

    mkdir -p $exp_dir/${task}_cmvn
    cp -r $exp_dir/$task/* $exp_dir/${task}_cmvn
    python feature_normalization.py -c $model_dir/config_feature.json \
        -T $exp_dir/${task}_cmvn -F $feat_dir/${task}_cmvn \
        -S ${model_dir}/stats.pt \
        -K "mel"            
fi

# Extracing token feature
if [ $stage -le 2 -a $stage_end -ge 2 ]; then
    python inference_bnf.py \
        -c $model_dir/config_vqvae.json \
        -d $exp_dir/${task}_cmvn \
        -o $exp_dir/${task}_token \
        -K "mel-${bnf_name}" \
        -m ${model_dir}/model.pt -g $gpu \
        --output_txt true

    cp $exp_dir/${task}_token/* $data_dir/
fi

if [ $clean_up == true ]; then
    rm -r $exp_dir
    rm -r $feat_dir
fi

echo "Finish extracting VQVAE bottleneck features"
