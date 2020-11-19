#!/bin/bash

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
BPEROOT=subword-nmt


SRC=en
TGT=fr
LANG=en-fr
ORIG='.'
DICTDIR = '../../../data-bin/wmt14.en-fr.newstest2014'


echo "tokenize."
cat $ORIG/test.$LANG.$SRC.txt | perl $TOKENIZER -threads 8 -l $SRC > $ORIG/test.$LANG.$SRC.out
cat $ORIG/test.$LANG.$TGT.txt | perl $TOKENIZER -threads 8 -l $TGT > $ORIG/test.$LANG.$TGT.out

echo "bpe."
python $BPEROOT/apply_bpe.py -c bpecodes < $ORIG/test.$LANG.$SRC.out > $ORIG/test.$LANG.$SRC
python $BPEROOT/apply_bpe.py -c bpecodes < $ORIG/test.$LANG.$TGT.out > $ORIG/test.$LANG.$TGT


fairseq-preprocess --source-lang $SRC --target-lang $TGT \
    --srcdict dict.$SRC.txt --tgtdict  dict.$TGT.txt --testpref $ORIG/test.$LANG  \
    --destdir $ORIG

echo "cleaning."
rm $ORIG/test.$LANG.$SRC.out
rm $ORIG/test.$LANG.$TGT.out
rm $ORIG/test.$LANG.$SRC
rm $ORIG/test.$LANG.$TGT






