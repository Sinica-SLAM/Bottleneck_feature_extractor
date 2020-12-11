export KALDI_ROOT=`pwd`/../kaldi
[ -d utils ] || ln -s $KALDI_ROOT/egs/wsj/s5/utils utils
# [ -d steps ] || ln -s $KALDI_ROOT/egs/wsj/s5/steps steps

# export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
# [ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
# . $KALDI_ROOT/tools/config/common_path.sh
# export LC_ALL=C

# . /home/roland/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate rol_py36_t14