#!/bin/bash

. ./cmd.sh
. ./path.sh

# make training features
#if [ $stage -eq 7 ]; then
  mfccdir=mfcc
  corpora="ami_ihm"
  for c in $corpora; do
    (
     data=data/$c/train
     steps/make_mfcc.sh --mfcc-config conf/mfcc.conf \
       --cmd "$train_cmd" --nj 80 \
       $data exp/make_mfcc/$c/train || touch $data/.error
     steps/compute_cmvn_stats.sh \
       $data exp/make_mfcc/$c/train || touch $data/.error
    ) &
  done
  wait
  if [ -f $data/.error ]; then
     rm $data/.error || true
     echo "Fail to extract features." && exit 1;
  fi
#fi
