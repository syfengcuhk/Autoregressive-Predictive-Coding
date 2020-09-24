#!/bin/bash

echo "$0: prepare raw data from $1 to $2"

. ./path.sh
apply-cmvn --utt2spk=ark:$1/utt2spk scp:$1/cmvn.scp scp:$1/feats.scp ark,t:$2/${3}
