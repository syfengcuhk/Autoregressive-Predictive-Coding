import os
import logging
import argparse
from collections import namedtuple

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils import data
import tensorboard_logger
from tensorboard_logger import log_value

from apc_model import APCModel
from datasets import LibriSpeech


PrenetConfig = namedtuple(
  'PrenetConfig', ['input_size', 'hidden_size', 'num_layers', 'dropout'])
RNNConfig = namedtuple(
  'RNNConfig', ['input_size', 'hidden_size', 'num_layers', 'dropout', 'residual'])


def main():
  parser = argparse.ArgumentParser(description="Configuration for extracting features based on a pretrained APC model")

  # Prenet architecture (note that all APC models in the paper DO NOT incoporate a prenet)
  parser.add_argument("--prenet_num_layers", default=0, type=int, help="Number of ReLU layers as prenet")
  parser.add_argument("--prenet_dropout", default=0., type=float, help="Dropout for prenet")

  # RNN architecture
  parser.add_argument("--rnn_num_layers", default=3, type=int, help="Number of RNN layers in the APC model")
  #parser.add_argument("--rnn_hidden_size", default=512, type=int, help="Number of hidden units in each RNN layer")
  parser.add_argument("--rnn_hidden_size", default=100, type=int, help="Number of hidden units in each RNN layer")
  parser.add_argument("--rnn_dropout", default=0., type=float, help="Dropout for each RNN output layer except the last one")
  parser.add_argument("--rnn_residual", action="store_true", help="Apply residual connections between RNN layers if specified")

  # Training configuration
  parser.add_argument("--optimizer", default="adam", type=str, help="The gradient descent optimizer (e.g., sgd, adam, etc.)")
  parser.add_argument("--batch_size", default=1, type=int, help="feed forward minibatch size")
  parser.add_argument("--learning_rate", default=0.0001, type=float, help="Initial learning rate")
  parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
  parser.add_argument("--time_shift", default=1, type=int, help="Given f_{t}, predict f_{t + n}, where n is the time_shift")
  parser.add_argument("--clip_thresh", default=1.0, type=float, help="Threshold for clipping the gradients")

  # Misc configurations
  parser.add_argument("--feature_dim", default=80, type=int, help="The dimension of the input frame")
  parser.add_argument("--rev_apc", action="store_true", help="Apply time-reversed feature extraction if specified")
  parser.add_argument("--load_data_workers", default=2, type=int, help="Number of parallel data loaders")
#  parser.add_argument("--experiment_name", default="foo", type=str, help="Name of this experiment")
  #parser.add_argument("--model_path", default="exp/clean-100/apc_3L512_feat80_e10_tshift1.dir", type=str, help="where to restore the pretrained model")
  parser.add_argument("--model_path", default="exp/clean-100/apc_3L100_feat80_e15_tshift1.dir", type=str, help="where to restore the pretrained model")
  #parser.add_argument("--model_name", default="apc_3L512_feat80_e15_tshift1__epoch_10.model", type=str, help="name of  the pretrained model")
  parser.add_argument("--model_name", default="apc_3L100_feat80_e15_tshift1__epoch_15.model", type=str, help="name of  the pretrained model")
#  parser.add_argument("--store_path", default="./logs", type=str, help="Where to save the trained models and logs")
  #parser.add_argument("--input_data", default="./librispeech_data/preprocessed_no_pad/dev-clean_subset", type=str, help="Path to the input data directory")
  parser.add_argument("--input_data", default="./librispeech_data/preprocessed_no_pad/dev-clean", type=str, help="Path to the input data directory")
  parser.add_argument("--bn_size", default=40, type=int, help="Number of hidden units in the extra BN layer")
  parser.add_argument("--bn_layer", action="store_true", help="Apply a top bn layer if specified") 
  parser.add_argument("--which_hid_to_output", default="-1", type=int, help="-1 for top hidden layer, -2 for 2nd-top hidden layer")
  #parser.add_argument("--rel_output_data_path", default="extracted_feats_no_pad/dev-clean_subset", type=str, help="Path to the output data directory")
  parser.add_argument("--rel_output_data_path", default="extracted_feats_no_pad/dev-clean", type=str, help="Path to the output data directory")
  config = parser.parse_args()
  model_dir = config.model_path
  model_dir_name = os.path.join(model_dir, config.model_name) 
  abs_output_data_path = os.path.join(model_dir, config.rel_output_data_path)
  abs_output_data_path_hid_layer = abs_output_data_path + "_"  + str(config.which_hid_to_output) 
  print(abs_output_data_path_hid_layer)
  os.makedirs(abs_output_data_path_hid_layer, exist_ok=True)

  logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(abs_output_data_path_hid_layer, 'log.txt'),
    filemode='w')

  # define a new Handler to log to console as well
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)

#  # load pretrained model:
#  model = APCModel()
#  model.load_state_dict(torch.load(model_dir))
#  device = torch.device("cuda")
#  model.to(device)
#  model.eval()


  logging.info('Model Parameters: ')
  logging.info('Prenet Depth: %d' % (config.prenet_num_layers))
  logging.info('Prenet Dropout: %f' % (config.prenet_dropout))
  logging.info('RNN Depth: %d ' % (config.rnn_num_layers))
  logging.info('RNN Hidden Dim: %d' % (config.rnn_hidden_size))
  logging.info('RNN Residual Connections: %s' % (config.rnn_residual))
  logging.info('RNN Dropout: %f' % (config.rnn_dropout))
  logging.info('Extra BN layer: %s ' % (config.bn_layer))
  if config.bn_layer == True:
    logging.info('Extra BN layer dim: %s ' % (config.bn_size))
#  logging.info('Optimizer: %s ' % (config.optimizer))
#  logging.info('Batch Size: %d ' % (config.batch_size))
#  logging.info('Initial Learning Rate: %f ' % (config.learning_rate))
  logging.info('Time Shift: %d' % (config.time_shift))
  logging.info('Gradient Clip Threshold: %f' % (config.clip_thresh))

  if config.prenet_num_layers == 0:
    prenet_config = None
    rnn_config = RNNConfig(
      config.feature_dim, config.rnn_hidden_size, config.rnn_num_layers,
      config.rnn_dropout, config.rnn_residual)
  else:
    prenet_config = PrenetConfig(
      config.feature_dim, config.rnn_hidden_size, config.prenet_num_layers,
      config.prenet_dropout)
    rnn_config = RNNConfig(
      config.rnn_hidden_size, config.rnn_hidden_size, config.rnn_num_layers,
      config.rnn_dropout, config.rnn_residual)

  model = APCModel(
    mel_dim=config.feature_dim,
    prenet_config=prenet_config,
    rnn_config=rnn_config,
    if_bn_layer=config.bn_layer).cuda()
  print("Restore model: "  + model_dir_name)
  model.load_state_dict(torch.load(model_dir_name))
  model.eval()

  # Load input data
  input_data = LibriSpeech(config.input_data)
  print("Loaded input data from " + config.input_data + "...") 
  input_data_loader = data.DataLoader(
    input_data, batch_size=config.batch_size, num_workers=config.load_data_workers, shuffle=False)
  with torch.set_grad_enabled(False):
    i = 0;
    for batch_x, batch_l in input_data_loader:
      batch_x = batch_x.cuda()
      batch_l = batch_l.cuda()
#      print("\nbatch_x of shape:")
#      print(list(batch_x.size()))
#      print("batch_l of shape:" + list(batch_l.size()))
      if config.bn_layer == False:
        if config.rev_apc == False:
          _, batch_feats, _ = model.forward(batch_x, batch_l)
#          hid_dim = config.rnn_hidden_size
        else:
#          print("feed forward data in time-reversed way")
          time_ids = [ time_id for time_id in range(batch_x.size(1)-1, -1, -1) ]
          rev_batch_x = batch_x[:, time_ids, :]
          _, rev_batch_feats, _ = model.forward(rev_batch_x, batch_l)
          batch_feats = rev_batch_feats[:, :, time_ids, :]
        hid_dim = config.rnn_hidden_size  
      else:# bn_layer on top, 3rd output is bn reps
        _, _, batch_feats = model.forward(batch_x, batch_l)
        hid_dim = config.bn_size
#      print("extracted features of shape:")
#      print(list(batch_feats.size()))
#      print("current file is:")
      utt_id = input_data_loader.dataset.ids[i]
#      print(utt_id)
      utt_id_length = input_data_loader.dataset.lengths[utt_id]
#      print("current file length is:")
#      print(utt_id_length)
#      print("save to .pt files in " + abs_output_data_path_hid_layer)
#      hid_dim = config.rnn_hidden_size
      if config.bn_layer == False:
          assert config.rnn_hidden_size == batch_feats.size(-1), "RNN hidden layer size in config does not equal to model config"
      else:
          assert config.bn_size == batch_feats.size(-1),  "BN layer sieze in config does not equal to model config"
      # batch_feats of shape [#num_hid_layers, bs, seq_len, hid_dim]
      feat_to_store = batch_feats[config.which_hid_to_output, :, :, :].view(-1, hid_dim)
      #torch.save(batch_feats[config.which_hid_to_output, :, :, :], os.path.join(abs_output_data_path_hid_layer, utt_id ))
      torch.save(feat_to_store, os.path.join(abs_output_data_path_hid_layer, utt_id ))
      if i % 1000 == 0:
        print("finished " + str(i) + " files")
      i = i + 1


#  val_set = LibriSpeech(os.path.join(config.librispeech_path, 'dev-clean'))
#  val_data_loader = data.DataLoader(
#    val_set, batch_size=config.batch_size, num_workers=config.load_data_workers, shuffle=False)

#  torch.save(model.state_dict(),
#    open(os.path.join(model_dir, config.experiment_name + '__epoch_0.model'), 'wb'))
#
#  global_step = 0
#  for epoch_i in range(config.epochs):
#
#    ####################
#    ##### Training #####
#    ####################
#
#    model.train()
#    train_losses = []
#    for batch_x, batch_l in train_data_loader:
#
#      _, indices = torch.sort(batch_l, descending=True)
#
#      batch_x = Variable(batch_x[indices]).cuda()
#      batch_l = Variable(batch_l[indices]).cuda()
#
#      outputs, _ = model(
#        batch_x[:, :-config.time_shift, :], batch_l - config.time_shift)
#
#      optimizer.zero_grad()
#      loss = criterion(outputs, batch_x[:, config.time_shift:, :])
#      train_losses.append(loss.item())
#      loss.backward()
#      grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_thresh)
#      optimizer.step()
#
#      log_value("training loss (step-wise)", float(loss.item()), global_step)
#      log_value("gradient norm", grad_norm, global_step)
#
#      global_step += 1
#
#    ######################
#    ##### Validation #####
#    ######################
#
#    model.eval()
#    val_losses = []
#    with torch.set_grad_enabled(False):
#      for val_batch_x, val_batch_l in val_data_loader:
#        _, val_indices = torch.sort(val_batch_l, descending=True)
#
#        val_batch_x = Variable(val_batch_x[val_indices]).cuda()
#        val_batch_l = Variable(val_batch_l[val_indices]).cuda()
#
#        val_outputs, _ = model(
#          val_batch_x[:, :-config.time_shift, :], val_batch_l - config.time_shift)
#
#        val_loss = criterion(val_outputs, val_batch_x[:, config.time_shift:, :])
#        val_losses.append(val_loss.item())
#
#    logging.info('Epoch: %d Training Loss: %.5f Validation Loss: %.5f' % (epoch_i + 1, np.mean(train_losses), np.mean(val_losses)))
#
#    log_value("training loss (epoch-wise)", np.mean(train_losses), epoch_i)
#    log_value("validation loss (epoch-wise)", np.mean(val_losses), epoch_i)
#
#    torch.save(model.state_dict(),
#      open(os.path.join(model_dir, config.experiment_name + '__epoch_%d' % (epoch_i + 1) + '.model'), 'wb'))


if __name__ == '__main__':
  main()
