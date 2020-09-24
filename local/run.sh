#!/bin/bash

exp_suffix= #e.g. _zr17, or blank if librispeech
stage=0
stop_stage=1
resume_epoch=0
prenet_num_layers=0
rnn_num_layers=3
rnn_hid_size=100
# if want residual, --rnn_residual
use_rev_apc=false

bs=32
lr=0.0001
epochs=40
model_id=40
time_shift=1
clip_thresh=1.0
add_residual=true
add_bn_layer=false
train_set=unlab_600_full #or clean-360 or other-500
dev_set=unlab_600_full #or clean-360 or other-500
feat_type=mfcc_cm
feat_size=13
nj=2
which_layer_to_extract=-1 # -1 for last hid layer, -2 for second last
model_used_to_extract=
abx_eval_set_appoint=dev-clean
#experiment_name=???
#store_path=???
librispeech_dir=libri-light_data/preprocessed
train_dev_dir_to_dump=libri-light_data/preprocessed/merge_tr_cv_without_pad
train_dev_name_to_dump=train-unlab_600_subset3600utt # or train-unlab_600_subset7200utt or just trian-unlab_600
test_name_to_dump=test_ark
. ./utils/parse_options.sh
if $add_residual; then
    flag_residual="--rnn_residual"
    suffix_residual="_res"
else
    flag_residual=
    suffix_residual=
fi
if $use_rev_apc; then
  flag_rev_apc="--rev_apc"
  prefix_rev_apc="rev_"
else
  flag_rev_apc=
  prefix_rev_apc=
fi
if $add_bn_layer; then
   flag_bn_layer="--bn_layer"
   suffix_bn="_bnf"
else
   flag_bn_layer=
   suffix_bn=
fi
if [ ! $prenet_num_layers == 0 ]; then
  prefix_prenet=_pre${prenet_num_layers}
else
  prefix_prenet=
fi
store_path=exp${exp_suffix}/$train_set
experiment_name=${prefix_rev_apc}apc${prefix_prenet}_${rnn_num_layers}L${rnn_hid_size}_feat${feat_size}_${feat_type}_e${epochs}_tshift${time_shift}${suffix_bn}${suffix_residual}
# model_dir is $store_path/${experiment_name}.dir
if [ $stage -le 0 ] && [ $stop_stage -gt 0 ]; then
  source /scratch/siyuanfeng/software/anaconda3/bin/activate apc  
#  mkdir -p $store_path/$experiment_name || exit 1;
  echo "$0: stage 0, train APC model"
  echo "$0: $(hostname)"
  python3 train_apc.py --rnn_num_layers $rnn_num_layers \
                       --prenet_num_layers $prenet_num_layers \
                       --rnn_hidden_size  $rnn_hid_size \
                       --librispeech_path $librispeech_dir \
                       --time_shift $time_shift --learning_rate $lr  \
                       --dev_set $dev_set --train_set $train_set --feature_dim $feat_size \
                       --load_data_workers $nj --store_path $store_path \
  	               $flag_residual  --experiment_name $experiment_name \
                       $flag_bn_layer $flag_rev_apc \
                       --resume_model_epoch_id $resume_epoch \
                       --batch_size $bs --epochs $epochs || exit 1;
  
  echo "$0: succeeded APC training..."
  source /scratch/siyuanfeng/software/anaconda3/bin/deactivate 
fi

if [ -z "$model_used_to_extract"  ]   ; then
  #use final epoch model
  extracted_folder_suffix=
  epoch_for_extraction=$epochs
else
  #use intermediate epoch model
  extracted_folder_suffix=_epoch${model_used_to_extract}
  epoch_for_extraction=$model_used_to_extract
fi
if [ $stage -le 1 ] && [ $stop_stage -gt 1 ]; then
  source /scratch/siyuanfeng/software/anaconda3/bin/activate apc
  echo "$0: stage 1, extract features from APC model"
  echo "$0: $(hostname)"
  if [ -z "$model_used_to_extract"  ]   ; then
    #use final epoch model
    extracted_folder_suffix=
    epoch_for_extraction=$epochs
  else
    #use intermediate epoch model
    extracted_folder_suffix=_epoch${model_used_to_extract}
    epoch_for_extraction=$model_used_to_extract
  fi
  if [ ! -f ${store_path}/${experiment_name}.dir/${experiment_name}__epoch_${epoch_for_extraction}.model ]; then
      echo "In direcotyr ${store_path}/${experiment_name}.dir"
      echo "Specified model (epoch:${epoch_for_extraction} for extraction does not exist."
      exit 1;
  fi
  for set in dev-clean dev-other test-clean test-other; do
    python3 extract_apc_feat.py --feature_dim $feat_size \
               --rnn_num_layers $rnn_num_layers --rnn_hidden_size  $rnn_hid_size $flag_residual \
               --batch_size 1 --epochs $epochs --time_shift $time_shift \
               $flag_bn_layer --learning_rate $lr --load_data_workers $nj $flag_rev_apc \
               --prenet_num_layers $prenet_num_layers \
               --model_path ${store_path}/${experiment_name}.dir \
               --model_name ${experiment_name}__epoch_${epoch_for_extraction}.model \
               --input_data $librispeech_dir/test/${set} \
               --which_hid_to_output $which_layer_to_extract \
               --rel_output_data_path extracted_feats_no_pad${extracted_folder_suffix}/${set}  || exit 1;
  done 
  source /scratch/siyuanfeng/software/anaconda3/bin/deactivate
fi

if   [ $stage -le 5 ] && [ $stop_stage -gt 5 ]; then
   echo "$0: stage 5, evaluated by ABX"
   echo "$0: $(hostname)"
   #conda activate libri-light
   echo "enter (libri-light)"
   source /scratch/siyuanfeng/software/anaconda3/bin/activate libri-light  # enter env: (libri-light)
   module load cuda/10.0
   PYTHONPATH=    # otherwise torchaudio cannot be imported
   echo "reset PYTHONPATH, value=$PYTHONPATH"
   libri_light_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/
   #for eval_set in dev-clean dev-other test-clean test-other; do
   #for eval_set in dev-clean ; do
    eval_set=$abx_eval_set_appoint
     #source_dir=$exp_dir/eval/z1_meanvar/test_by_utt/${eval_set}_npy/
     #source_dir=/scratch/siyuanfeng/software/Autoregressive-Predictive-Coding/${store_path}/${experiment_name}.dir/extracted_feats_no_pad_${model_id}/${eval_set}_${which_layer_to_extract} #-1 means last hidden layer representation of APC
     source_dir=${store_path}/${experiment_name}.dir/extracted_feats_no_pad${extracted_folder_suffix}/${eval_set}_${which_layer_to_extract} #-1 means last hidden layer representation of APC
     #target_dir=$exp_dir/eval/z1_meanvar/test_by_utt/eval/$eval_set
     #target_dir=/scratch/siyuanfeng/software/Autoregressive-Predictive-Coding/${store_path}/${experiment_name}.dir/extracted_feats_no_pad_${model_id}/eval/${eval_set}_${which_layer_to_extract}
     target_dir=${store_path}/${experiment_name}.dir/extracted_feats_no_pad${extracted_folder_suffix}/eval/${eval_set}_${which_layer_to_extract}
     echo "$0: target dir: $target_dir"
     mkdir -p $target_dir || exit 1;
     #reason use python3 instead of python is that in ~/.bash_profile I add alias python=@/anaconda3/bin/python, not in @anaconda3/envs/libri-light/bin/python
     ls $source_dir/*.pt | wc -l
     python3 ${libri_light_root}/eval/eval_ABX.py $source_dir/ ${libri_light_root}/eval/ABX_src/ABX_data/${eval_set}.item --file_extension .pt --out $target_dir/ --cuda

   #done
   #conda deactivate
   source /scratch/siyuanfeng/software/anaconda3/bin/deactivate # reset
   module load cuda/8.0 # reset
   #. ./path.sh
   PYTHONPAT=./:./src:$KALDI_PYTHON_DIR:$PYTHONPATH # reset
fi

if [ $stage -le 6 ] && [ $stop_stage -gt 6 ]; then
  source /scratch/siyuanfeng/software/anaconda3/bin/activate apc
  echo "$0: stage 6, extract features of training data from APC model as kaldi format for kaldi use"
  echo "$0: $(hostname)"
    python3 extract_apc_feat_to_kaldi_format.py --feature_dim $feat_size \
               --rnn_num_layers $rnn_num_layers --rnn_hidden_size  $rnn_hid_size $flag_residual \
               --batch_size 1 --epochs $epochs --time_shift $time_shift \
               $flag_bn_layer --learning_rate $lr --load_data_workers $nj $flag_rev_apc \
               --prenet_num_layers $prenet_num_layers \
               --model_path ${store_path}/${experiment_name}.dir \
               --model_name ${experiment_name}__epoch_${epochs}.model \
               --input_data $train_dev_dir_to_dump/$train_dev_name_to_dump \
               --which_hid_to_output $which_layer_to_extract \
               --rel_output_data_path extracted_feats_no_pad${extracted_folder_suffix}/$train_dev_name_to_dump  || exit 1;
  source /scratch/siyuanfeng/software/anaconda3/bin/deactivate
fi

. ./path.sh
if [ $stage -le 7 ] && [ $stop_stage -gt 7 ]; then
  echo "construct kaldi data dir using utils/copy_data_dir.sh"
  source_dir=${store_path}/${experiment_name}.dir/extracted_feats_no_pad${extracted_folder_suffix}/${train_dev_name_to_dump}_${which_layer_to_extract}
  utils/copy_data_dir.sh /tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/data/train_${train_set} $source_dir/data_for_kaldi || exit 1   
  mkdir -p $source_dir/data_for_kaldi/.backup || exit 1;
  mv $source_dir/data_for_kaldi/{feats.scp,cmvn.scp} $source_dir/data_for_kaldi/.backup/
  copy-feats ark:$source_dir/feats.ark ark,scp:$source_dir/data_for_kaldi/feats.ark,$source_dir/data_for_kaldi/feats.scp || exit 1
  steps/compute_cmvn_stats.sh $source_dir/data_for_kaldi/ || exit 1;
  utils/fix_data_dir.sh $source_dir/data_for_kaldi || exit 1; # sort feats.scp
fi

if [ $stage -le 8 ] && [ $stop_stage -gt 8 ]; then
  source /scratch/siyuanfeng/software/anaconda3/bin/activate apc
  echo "$0: stage 8, extract features of test data from APC model as kaldi format for kaldi use"
  echo "$0: $(hostname)"
#    for set in dev-clean dev-other test-clean test-other; do
  eval_set=$abx_eval_set_appoint
  python3 extract_apc_feat_to_kaldi_format.py --feature_dim $feat_size \
             --rnn_num_layers $rnn_num_layers --rnn_hidden_size  $rnn_hid_size $flag_residual \
             --batch_size 1 --epochs $epochs --time_shift $time_shift \
             $flag_bn_layer --learning_rate $lr --load_data_workers $nj $flag_rev_apc \
             --prenet_num_layers $prenet_num_layers \
             --model_path ${store_path}/${experiment_name}.dir \
             --model_name ${experiment_name}__epoch_${epochs}.model \
             --input_data $librispeech_dir/test/${eval_set} \
             --which_hid_to_output $which_layer_to_extract \
             --rel_output_data_path extracted_feats_no_pad${extracted_folder_suffix}/$test_name_to_dump/$eval_set  || exit 1;
#    done
  source /scratch/siyuanfeng/software/anaconda3/bin/deactivate
fi

if [ $stage -le 9 ] && [ $stop_stage -gt 9 ]; then
  echo "construct kaldi data dir using utils/copy_data_dir.sh"
  eval_set=$abx_eval_set_appoint
  source_dir=${store_path}/${experiment_name}.dir/extracted_feats_no_pad${extracted_folder_suffix}/$test_name_to_dump/${eval_set}_${which_layer_to_extract}
  utils/copy_data_dir.sh /tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/data/$eval_set $source_dir/data_for_kaldi || exit 1
  mkdir -p $source_dir/data_for_kaldi/.backup || exit 1;
  mv $source_dir/data_for_kaldi/{feats.scp,cmvn.scp} $source_dir/data_for_kaldi/.backup/
  copy-feats ark:$source_dir/feats.ark ark,scp:$source_dir/data_for_kaldi/feats.ark,$source_dir/data_for_kaldi/feats.scp || exit 1
  steps/compute_cmvn_stats.sh $source_dir/data_for_kaldi/ || exit 1;
fi

if [ $stage -le 10 ] && [ $stop_stage -gt 10 ];then
 echo "cleaning up data"
 dir_to_clean=${store_path}/${experiment_name}.dir/extracted_feats_no_pad${extracted_folder_suffix}/
 for set in dev-clean dev-other test-clean test-other; do
  rm -rf $dir_to_clean/$set
 done

fi

exit

