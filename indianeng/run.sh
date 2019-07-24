#!/bin/bash

# Copyright 2016  Allen Guo
#           2017  Xiaohui Zhang
# Apache 2.0

. ./cmd.sh
. ./path.sh

# paths to corpora (see below for example)
ami=/home/nijil/kaldi/egs/multi_en/s5/vault/ami
fisher=/home/nijil/kaldi/egs/multi_en/s5/vault/fisher
librispeech=/home/nijil/kaldi/egs/multi_en/s5/vault
swbd=/home/nijil/kaldi/egs/multi_en/s5/vault/switchboard
nptel=/home/nijil/kaldi/egs/multi_en/s5/vault/nptel


set -e
# check for kaldi_lm
#which get_word_map.pl > /dev/null
#if [ $? -ne 0 ]; then
#  echo "This recipe requires installation of tools/kaldi_lm. Please run extras/kaldi_lm.sh in tools/" && exit 1;
#fi

# general options
stage=$1
cleanup_stage=1
multi=multi_a  # This defines the "variant" we're using; see README.md
srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"

. utils/parse_options.sh

.
# and train a G2P model using the combined lexicon
# in data/local/dict_combined
if [ $stage -eq 1 ]; then
  # We prepare the basic dictionary in data/local/dict_combined.
  local/prepare_dict.sh $swbd $tedlium2
  (
   steps/dict/train_g2p_phonetisaurus.sh --stage 0 --silence-phones \
     "data/local/dict_combined/silence_phones.txt" data/local/dict_combined/lexicon.txt exp/g2p || touch exp/g2p/.error
  ) &
fi

# Prepare corpora data
if [ $stage -eq 2 ]; then
  mkdir -p data/local
  # fisher
  local/fisher_data_prep.sh $fisher
  utils/fix_data_dir.sh data/fisher/train
  # swbd
  local/swbd1_data_prep.sh $swbd
  utils/fix_data_dir.sh data/swbd/train
  # librispeech
  local/librispeech_data_prep.sh $librispeech/LibriSpeech/train-clean-100 data/librispeech_100/train
  local/librispeech_data_prep.sh $librispeech/LibriSpeech/train-clean-360 data/librispeech_360/train
  local/librispeech_data_prep.sh $librispeech/LibriSpeech/train-other-500 data/librispeech_500/train
  local/librispeech_data_prep.sh $librispeech/LibriSpeech/test-clean data/librispeech/test
  # tedlium
  local/tedlium_prepare_data.sh $tedlium2


  #AMI 
  local/ami_text_prep.sh data/local/ami/annotations/
  local/ami_ihm_data_prep.sh vault/ami/wav_db/

  # wsj
  #local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?
  #local/wsj_format_data.sh
  #utils/copy_data_dir.sh --spk_prefix wsj_ --utt_prefix wsj_ data/wsj/train_si284 data/wsj/train
  #rm -r data/wsj/train_si284 2>/dev/null || true
  # hub4_en
  #local/hub4_en_data_prep.sh $hub4_en_96 $hub4_en_97
  # eval2000 (test)
  #local/eval2000_data_prep.sh $eval2000
  #utils/fix_data_dir.sh data/eval2000/test
  # rt03 (test)
  #local/rt03_data_prep.sh $rt03
  #utils/fix_data_dir.sh data/rt03/test
fi

# Normalize transcripts
if [ $stage -eq 3 ]; then
  for f in data/*/{train,test}/text; do
    echo Normalizing $f
    cp $f $f.orig
    local/normalize_transcript.py $f.orig > $f
  done
fi

# Synthesize pronounciations for OOV words across all training transcripts and produce the final lexicon.
if [ $stage -eq 4 ]; then
  wait # Waiting for train_g2p.sh to finish
  if [ -f exp/g2p/.error ]; then
     rm exp/g2p/.error || true
     echo "Fail to train the G2P model." && exit 1;
  fi
  dict_dir=data/local/dict_nosp
  mkdir -p $dict_dir
  rm $dict_dir/lexiconp.txt 2>/dev/null || true
  cp data/local/dict_combined/{extra_questions,nonsilence_phones,silence_phones,optional_silence}.txt $dict_dir

  echo 'Gathering missing words...'
  
  lexicon=data/local/dict_combined/lexicon.txt
  g2p_tmp_dir=data/local/g2p_phonetisarus
  mkdir -p $g2p_tmp_dir

  # awk command from http://stackoverflow.com/questions/2626274/print-all-but-the-first-three-columns
  cat data/*/train/text | \
    local/count_oovs.pl $lexicon | \
    awk '{if (NF > 3 ) {for(i=4; i<NF; i++) printf "%s ",$i; print $NF;}}' | \
    perl -ape 's/\s/\n/g;' | \
    sort | uniq > $g2p_tmp_dir/missing.txt
  cat $g2p_tmp_dir/missing.txt | \
    grep "^[a-z]*$"  > $g2p_tmp_dir/missing_onlywords.txt

  steps/dict/apply_g2p_phonetisaurus.sh --nbest 1 $g2p_tmp_dir/missing_onlywords.txt exp/g2p exp/g2p/oov_lex || exit 1;
  cp exp/g2p/oov_lex/lexicon.lex $g2p_tmp_dir/missing_lexicon.txt

  extended_lexicon=$dict_dir/lexicon.txt
  echo "Adding new pronunciations to get extended lexicon $extended_lexicon"
  cat <(cut -f 1,3 $g2p_tmp_dir/missing_lexicon.txt) $lexicon | sort | uniq > $extended_lexicon
fi

# We'll do multiple iterations of pron/sil-prob estimation. So the structure of
# the dict/lang dirs are designed as ${dict/lang_root}_${dict_affix}, where dict_affix
# is "nosp" or the name of the acoustic model we use to estimate pron/sil-probs.
dict_root=data/local/dict
lang_root=data/lang

# prepare (and validate) lang directory
if [ $stage -eq 5 ]; then
  utils/prepare_lang.sh ${dict_root}_nosp "<unk>" data/local/tmp/lang_nosp ${lang_root}_nosp
fi

# prepare LM and test lang directory
if [ $stage -eq 6 ]; then
  mkdir -p data/local/lm
  cat data/{fisher,swbd}/train/text > data/local/lm/text
  local/train_lms.sh  # creates data/local/lm/3gram-mincount/lm_unpruned.gz
  utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
    ${lang_root}_nosp data/local/lm/3gram-mincount/lm_unpruned.gz \
    ${dict_root}_nosp/lexicon.txt ${lang_root}_nosp_fsh_sw1_tg
fi

# make training features
if [ $stage -eq 7 ]; then
  mfccdir=mfcc
  corpora="ami_ihm fisher librispeech_100 librispeech_360 librispeech_500 swbd tedlium"
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
fi

# fix and validate training data directories
if [ $stage -eq 8 ]; then
  # get rid of spk2gender files because not all corpora have them
  rm data/*/train/spk2gender 2>/dev/null || true
  # create reco2channel_and_file files for wsj and librispeech
  for c in librispeech_100 librispeech_360 librispeech_500; do
    awk '{print $1, $1, "A"}' data/$c/train/wav.scp > data/$c/train/reco2file_and_channel;
  done
  # apply standard fixes, then validate
  for f in data/*/train; do
    utils/fix_data_dir.sh $f
    utils/validate_data_dir.sh $f
  done
fi

# make test features
if [ $stage -eq 9 ]; then
  mfccdir=mfcc
  corpora="tedlium librispeech"
  for c in $corpora; do
    data=data/$c/test
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf \
      --cmd "$train_cmd" --nj 80 \
      $data exp/make_mfcc/$c/test || exit 1;
    steps/compute_cmvn_stats.sh \
      $data exp/make_mfcc/$c/test || exit 1;
  done
fi

# fix and validate test data directories
if [ $stage -eq 10 ]; then
  for f in data/*/test; do
    utils/fix_data_dir.sh $f
    utils/validate_data_dir.sh $f
  done
fi

# train mono on swbd 10k short (nodup)
if [ $stage -eq 11 ]; then
 local/make_partitions.sh --multi $multi --stage 1 || exit 1;
 steps/train_mono.sh --boost-silence 1.25 --nj 80 --cmd "$train_cmd" \
   data/$multi/mono ${lang_root}_nosp exp/$multi/mono || exit 1;
fi

# train tri1a and tri1b (first and second triphone passes) on swbd 30k (nodup)
if [ $stage -eq 12 ]; then
  local/make_partitions.sh --multi $multi --stage 2 || exit 1;
  steps/align_si.sh --boost-silence 1.25 --nj 80 --cmd "$train_cmd" \
    data/$multi/mono_ali ${lang_root}_nosp exp/$multi/mono exp/$multi/mono_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 3200 30000 \
    data/$multi/tri1a ${lang_root}_nosp exp/$multi/mono_ali exp/$multi/tri1a || exit 1;

  steps/align_si.sh --boost-silence 1.25 --nj 80 --cmd "$train_cmd" \
    data/$multi/tri1a_ali ${lang_root}_nosp exp/$multi/tri1a exp/$multi/tri1a_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 3200 30000 \
    data/$multi/tri1b ${lang_root}_nosp exp/$multi/tri1a_ali exp/$multi/tri1b || exit 1;
  # decode
  (  
    gmm=tri1b
    graph_dir=exp/$multi/$gmm/graph_tg
    utils/mkgraph.sh ${lang_root}_nosp_fsh_sw1_tg \
      exp/$multi/$gmm $graph_dir || exit 1;
    for e in eval2000 rt03; do
      steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" --config conf/decode.config $graph_dir \
        data/$e/test exp/$multi/$gmm/decode_tg_$e || exit 1;
    done
  )&
fi

# train tri2 (third triphone pass) on swbd 100k (nodup)
if [ $stage -eq 13 ]; then
 local/make_partitions.sh --multi $multi --stage 3 || exit 1;
 steps/align_si.sh --boost-silence 1.25 --nj 80 --cmd "$train_cmd" \
   data/$multi/tri1b_ali ${lang_root}_nosp exp/$multi/tri1b exp/$multi/tri1b_ali || exit 1;
 steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 5500 90000 \
   data/$multi/tri2 ${lang_root}_nosp exp/$multi/tri1b_ali exp/$multi/tri2 || exit 1;
fi

# train tri3a (4th triphone pass) on whole swbd
if [ $stage -eq 14 ]; then
  local/make_partitions.sh --multi $multi --stage 4 || exit 1;
  steps/align_si.sh --boost-silence 1.25 --nj 80 --cmd "$train_cmd" \
    data/$multi/tri2_ali ${lang_root}_nosp exp/$multi/tri2 exp/$multi/tri2_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 11500 200000 \
    data/$multi/tri3a ${lang_root}_nosp exp/$multi/tri2_ali exp/$multi/tri3a || exit 1;
  # decode
  (  
    gmm=tri3a
    graph_dir=exp/$multi/$gmm/graph_tg
    utils/mkgraph.sh ${lang_root}_nosp_fsh_sw1_tg \
      exp/$multi/$gmm $graph_dir || exit 1;
    for e in librispeech tedlium; do
      steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config $graph_dir \
        data/$e/test exp/$multi/$gmm/decode_tg_$e || exit 1;
    done
  )&
fi

# train tri3b (LDA+MLLT) on whole fisher + swbd (nodup)
if [ $stage -eq 15 ]; then
  local/make_partitions.sh --multi $multi --stage 5 || exit 1;
  steps/align_si.sh --boost-silence 1.25 --nj 100 --cmd "$train_cmd" \
    data/$multi/tri3a_ali ${lang_root}_nosp exp/$multi/tri3a exp/$multi/tri3a_ali || exit 1;
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 11500 400000 \
    data/$multi/tri3b ${lang_root}_nosp exp/$multi/tri3a_ali exp/$multi/tri3b || exit 1;
  # decode
  (  
    gmm=tri3b
    graph_dir=exp/$multi/$gmm/graph_tg
    utils/mkgraph.sh ${lang_root}_nosp_fsh_sw1_tg \
      exp/$multi/$gmm $graph_dir || exit 1;
    for e in librispeech tedlium; do
      steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config $graph_dir \
        data/$e/test exp/$multi/$gmm/decode_tg_$e || exit 1;
    done
  )&
fi

# reestimate pron & sil-probs
dict_affix=${multi}_tri3b
if [ $stage -eq 16 ]; then
  gmm=tri3b
  steps/get_prons.sh --cmd "$train_cmd" data/$multi/$gmm ${lang_root}_nosp exp/$multi/$gmm
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    ${dict_root}_nosp exp/$multi/$gmm/pron_counts_nowb.txt \
    exp/$multi/$gmm/sil_counts_nowb.txt exp/$multi/$gmm/pron_bigram_counts_nowb.txt ${dict_root}_${dict_affix}
  utils/prepare_lang.sh ${dict_root}_${dict_affix} "<unk>" data/local/lang_${dict_affix} ${lang_root}_${dict_affix}
  utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
    ${lang_root}_${dict_affix} data/local/lm/3gram-mincount/lm_unpruned.gz \
    ${dict_root}_${dict_affix}/lexicon.txt ${lang_root}_${dict_affix}_fsh_sw1_tg
  # decode
  (  
    gmm=tri3b
    graph_dir=exp/$multi/$gmm/graph_tg_sp
    utils/mkgraph.sh ${lang_root}_${dict_affix}_fsh_sw1_tg \
      exp/$multi/$gmm $graph_dir || exit 1;
    for e in librispeech; do
      steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config $graph_dir \
        data/$e/test exp/$multi/$gmm/decode_tg_sp_$e || exit 1;
    done
  )&
fi

lang=${lang_root}_${dict_affix}
if [ $stage -eq 17 ]; then
  # This does the actual data cleanup.
  steps/cleanup/clean_and_segment_data.sh --stage $cleanup_stage --nj 100 --cmd "$train_cmd" \
  data/tedlium/train $lang exp/$multi/tri3b exp/$multi/tri3b_tedlium_cleaning_work data/$multi/tedlium_cleaned/train
fi

# train tri4 on fisher + swbd + tedlium (nodup)
if [ $stage -eq 18 ]; then
  local/make_partitions.sh --multi $multi --stage 6 || exit 1;
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 \
    data/$multi/tri3b_ali $lang \
    exp/$multi/tri3b exp/$multi/tri3b_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" 11500 800000 \
    data/$multi/tri4 $lang exp/$multi/tri3b_ali exp/$multi/tri4 || exit 1;
  (  
    gmm=tri4
    graph_dir=exp/$multi/$gmm/graph_tg
    utils/mkgraph.sh ${lang}_fsh_sw1_tg \
      exp/$multi/$gmm $graph_dir || exit 1;
    for e in librispeech; do
      steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config $graph_dir \
        data/$e/test exp/$multi/$gmm/decode_tg_$e || exit 1;
    done
  )&
fi

#DID NOT ATTEMPT LACK OF DATASET
# train tri5a on fisher + swbd + tedlium + wsj + hub4_en (nodup)
if [ $stage -eq 19 ]; then
  local/make_partitions.sh --multi $multi --stage 7 || exit 1;
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 \
    data/$multi/tri4_ali $lang \
    exp/$multi/tri4 exp/$multi/tri4_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" 11500 1600000 \
    data/$multi/tri5a $lang exp/$multi/tri4_ali exp/$multi/tri5a || exit 1;
  (  
    gmm=tri5a
    graph_dir=exp/$multi/$gmm/graph_tg
    utils/mkgraph.sh ${lang}_fsh_sw1_tg \
      exp/$multi/$gmm $graph_dir || exit 1;
    for e in eval2000 rt03; do
      steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config $graph_dir \
        data/$e/test exp/$multi/$gmm/decode_tg_$e || exit 1;
    done
  )&
fi

# reestimate pron & sil-probs
dict_affix=${multi}_tri5a
if [ $stage -eq 20 ]; then
  gmm=tri4
  steps/get_prons.sh --cmd "$train_cmd" data/$multi/$gmm ${lang_root}_nosp exp/$multi/$gmm
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    ${dict_root}_nosp exp/$multi/$gmm/pron_counts_nowb.txt \
    exp/$multi/$gmm/sil_counts_nowb.txt exp/$multi/$gmm/pron_bigram_counts_nowb.txt ${dict_root}_${dict_affix}
  utils/prepare_lang.sh ${dict_root}_${dict_affix} "<unk>" data/local/lang_${dict_affix} ${lang_root}_${dict_affix}
  utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
    ${lang_root}_${dict_affix} data/local/lm/3gram-mincount/lm_unpruned.gz \
    ${dict_root}_${dict_affix}/lexicon.txt ${lang_root}_${dict_affix}_fsh_sw1_tg
  # re-decode after re-estimating sil & pron-probs
  (  
    gmm=tri4
    graph_dir=exp/$multi/$gmm/graph_tg_sp
    utils/mkgraph.sh ${lang_root}_${dict_affix}_fsh_sw1_tg \
      exp/$multi/$gmm $graph_dir || exit 1;
    for e in librispeech; do
      steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config $graph_dir \
        data/$e/test exp/$multi/$gmm/decode_tg_sp_$e || exit 1;
    done
  )&
fi

lang=${lang_root}_${dict_affix}
# train tri5b on fisher + swbd + tedlium + wsj + hub4_en + librispeeh460 (nodup)
if [ $stage -eq 21 ]; then
  local/make_partitions.sh --multi $multi --stage 8 || exit 1;
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 \
    data/$multi/tri4_ali $lang \
    exp/$multi/tri4 exp/$multi/tri5a_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" 11500 2000000 \
    data/$multi/tri5b $lang exp/$multi/tri5a_ali exp/$multi/tri5b || exit 1;
  (  
    gmm=tri5b
    graph_dir=exp/$multi/$gmm/graph_tg
    utils/mkgraph.sh ${lang}_fsh_sw1_tg \
      exp/$multi/$gmm $graph_dir || exit 1;
    for e in librispeech; do
      steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config $graph_dir \
        data/$e/test exp/$multi/$gmm/decode_tg_$e || exit 1;
    done
  )&
fi

# train tri6a on fisher + swbd + tedlium + wsj + hub4_en + librispeeh960 (nodup)
if [ $stage -eq 22 ]; then
  local/make_partitions.sh --multi $multi --stage 9 || exit 1;
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 \
    data/$multi/tri5b_ali $lang \
    exp/$multi/tri5b exp/$multi/tri5b_ali || exit 1;

  steps/train_sat.sh --cmd "$train_cmd" 14000 2400000 \
    data/$multi/tri6a $lang exp/$multi/tri5b_ali exp/$multi/tri6a || exit 1;
  (  
    gmm=tri6a
    graph_dir=exp/$multi/$gmm/graph_tg
    utils/mkgraph.sh ${lang}_fsh_sw1_tg \
      exp/$multi/$gmm $graph_dir || exit 1;
    for e in librispeech; do
      steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config $graph_dir \
        data/$e/test exp/$multi/$gmm/decode_tg_$e || exit 1;
    done
  )&
fi

echo "ALL DONE YAEY :) "

