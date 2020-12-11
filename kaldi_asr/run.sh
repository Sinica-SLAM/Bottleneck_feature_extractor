#!/usr/bin/env bash

stage=3
stage_end=3
nj=1

data_dir=data/test

bnf_name=prefinal-l
bnf_data_dir=data/${data_name}_ppg

exp_dir=exp
ivector_dir=exp/nnet3_cleaned/extractor
nnet_dir=exp/chain_cleaned/tdnn_1d_sp

. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e

if [ $stage -le 0 -a $stage_end -ge 0 ]; then
    echo "$0: get data"
    data_name=`basename $data_dir`
    utils/copy_data_dir.sh $data_dir $exp_dir/${data_name}_hires
    utils/data/resample_data_dir.sh 16000 $exp_dir/${data_name}_hires
fi

if [ $stage -le 1 -a $stage_end -ge 1 ]; then
    echo "$0: extracting 16k high-resolution MFCC features"
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" $exp_dir/${data_name}_hires || exit 1;
    steps/compute_cmvn_stats.sh $exp_dir/${data_name}_hires || exit 1;
    utils/fix_data_dir.sh $exp_dir/${data_name}_hires
fi

if [ $stage -le 2 -a $stage_end -ge 2 ]; then
    echo "$0: extracting i-vectors"
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
        $exp_dir/${data_name}_hires $ivector_dir \
        exp/nnet3_cleaned/ivectors_${data_name}_hires || exit 1;
fi

if [ $stage -le 3 -a $stage_end -ge 3 ]; then
    echo "$0: extracting phonetic representations"
    steps/nnet3/make_bottleneck_features.sh --cmd "$train_cmd" --nj $nj \
        --ivector_dir exp/nnet3_cleaned/ivectors_${data_name}_hires \
        $bnf_name \
        $exp_dir/${data_name}_hires \
        $bnf_data_dir \
        $nnet_dir
fi
