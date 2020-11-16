#!/bin/sh

###############################################################################
### How to run?
### Test cpu accuracy. Just run
### bash run_infernece_cpu_accuracy.sh
###############################################################################

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

BATCH_SIZE=256

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

MODEL=data-bin/wmt14.en-fr.joined-dict.transformer/model.pt
DATA=data-bin/wmt14.en-fr.joined-dict.newstest2014

fairseq-generate $DATA \
    --path $MODEL \
    --batch-size $BATCH_SIZE --beam 5 --remove-bpe \
    --cpu $ARGS --quiet
