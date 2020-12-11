#!/usr/bin/env bash

cmd=run.pl

stage=3
stop_stage=3
nj=1

bnf_name=prefinal-l
ivector_dir=exp/nnet3_cleaned/extractor
nnet_dir=exp/chain_cleaned/tdnn_1d_sp

if [ -f kaldi_asr/path.sh ]; then
    echo "Change the environment with kaldi_asr/path.sh"
    . kaldi_asr/path.sh
fi
. utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 <data-dir> <bnf-data-dir>"
  echo "e.g.: $0 data/train exp/train_bnf"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --bnf-name <prefinal-l>                          # bottlecneck layer name of the acoustic model"
  echo "                                                   # you can check the tdnnf.network.xconfig file"
  echo "  --ivector-dir <exp/nnet3_cleaned/extractor>      # ivector model dir."
  echo "  --nnet-dir <exp/chain_cleaned/tdnn_1d_sp>        # acoustic model dir."
  exit 1;
fi

data_dir=$1
bnf_data_dir=$2

# you might not want to do this for interactive shells.
set -e

if [ $stage -le 0 -a $stop_stage -ge 0 ]; then
    echo "$0: get data"
    data_name=`basename $data_dir`
    utils/copy_data_dir.sh $data_dir $bnf_data_dir/${data_name}_hires
    utils/data/resample_data_dir.sh 16000 $bnf_data_dir/${data_name}_hires
fi

if [ $stage -le 1 -a $stop_stage -ge 1 ]; then
    echo "$0: extracting 16k high-resolution MFCC features"
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
        --cmd "$cmd" $bnf_data_dir/${data_name}_hires || exit 1;
    steps/compute_cmvn_stats.sh $bnf_data_dir/${data_name}_hires || exit 1;
    utils/fix_data_dir.sh $bnf_data_dir/${data_name}_hires
fi

if [ $stage -le 2 -a $stop_stage -ge 2 ]; then
    echo "$0: extracting i-vectors"
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj $nj \
        $bnf_data_dir/${data_name}_hires $ivector_dir \
        $bnf_data_dir/${data_name}_hires_ivector || exit 1; 
fi

if [ $stage -le 3 -a $stop_stage -ge 3 ]; then
    echo "$0: extracting bottleneck features"
    steps/nnet3/make_bottleneck_features.sh --cmd "$cmd" --nj $nj \
        --ivector_dir $bnf_data_dir/${data_name}_hires_ivector \
        $bnf_name \
        $bnf_data_dir/${data_name}_hires \
        $(utils/make_absolute.sh $bnf_data_dir) \
        $nnet_dir
fi
