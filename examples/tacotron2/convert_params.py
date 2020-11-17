import torch
import numpy as np
import pickle

import os
from collections import OrderedDict

def convert_param(torch_model_dict, paddle_model_dict, source_weight_name, target_weight_name):
    if source_weight_name in torch_model_dict:
        source_weight = torch_model_dict[source_weight_name]

        target_weight = source_weight.data.cpu().numpy()
        print("{}: {} => {}: {}".format(source_weight_name, source_weight.shape,
                                        target_weight_name, target_weight.shape))
        paddle_model_dict[target_weight_name] = target_weight


def convert_embedding(torch_model_dict, paddle_model_dict, source_prefix, target_prefix):
    convert_param(torch_model_dict, paddle_model_dict, "{}.weight".format(source_prefix), "{}.weight".format(target_prefix))

def convert_conv1d(torch_model_dict, paddle_model_dict, source_prefix, target_prefix, bias=True):
    convert_param(torch_model_dict, paddle_model_dict, "{}.weight".format(source_prefix), "{}.weight".format(target_prefix))
    if bias:
        convert_param(torch_model_dict, paddle_model_dict, "{}.bias".format(source_prefix), "{}.bias".format(target_prefix))


def convert_batchnorm(torch_model_dict, paddle_model_dict, source_prefix, target_prefix):
    torchlist = ["weight", "bias", "running_mean", "running_var"]
    paddlelist = ["weight", "bias", "_mean", "_variance"]
    for torchname, paddlename in zip(torchlist, paddlelist):
        convert_param(torch_model_dict, paddle_model_dict, "{}.{}".format(source_prefix, torchname), "{}.{}".format(target_prefix, paddlename))

def convert_lstm(torch_model_dict, paddle_model_dict, source_prefix, target_prefix, forward=False, reverse=False):
    torchlist = ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]
    paddlelist = ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]
    for torchname, paddlename in zip(torchlist, paddlelist):
        if reverse:
            torchname += '_l0_reverse'
            paddlename += '_l0_reverse'
        if forward:
            torchname += '_l0'
            paddlename += '_l0'

        convert_param(torch_model_dict, paddle_model_dict, "{}.{}".format(source_prefix, torchname), "{}.{}".format(target_prefix, paddlename))

def convert_tacotron2_encoder(torch_model_dict, paddle_model_dict):
    convert_embedding(torch_model_dict, paddle_model_dict, "embedding", "embedding")
    for i in range(3):
        convert_conv1d(torch_model_dict, paddle_model_dict, "encoder.convolutions.{}.0.conv".format(i), "encoder.convolutions_{}.conv".format(i))
        convert_batchnorm(torch_model_dict, paddle_model_dict, "encoder.convolutions.{}.1".format(i), "encoder.batchnorms{}".format(i))
    convert_lstm(torch_model_dict, paddle_model_dict, "encoder.lstm", "encoder.lstm", forward=True)
    convert_lstm(torch_model_dict, paddle_model_dict, "encoder.lstm", "encoder.lstm", reverse=True)

def convert_linear(torch_model_dict, paddle_model_dict, source_prefix, target_prefix, bias=True):
    torchlist = ["linear_layer.weight", "linear_layer.bias"]
    paddlelist = ["weight", "bias"]
    for torchname, paddlename in zip(torchlist, paddlelist):
        source_weight_name = "{}.{}".format(source_prefix, torchname)
        target_weight_name = "{}.{}".format(target_prefix, paddlename)
        if source_weight_name in torch_model_dict:
            source_weight = torch_model_dict[source_weight_name]

            target_weight = source_weight.T.data.cpu().numpy()
            print("{}: {} => {}: {}".format(source_weight_name, source_weight.shape,
                                            target_weight_name, target_weight.shape))
            paddle_model_dict[target_weight_name] = target_weight
            if bias == False:
                break

def convert_tacotron2_decoder(torch_model_dict, paddle_model_dict):
    for i in range(2):
        convert_linear(torch_model_dict, paddle_model_dict, "decoder.prenet.layers.{}".format(i), "decoder.prenet.linear{}".format(i+1), bias=False)
    convert_lstm(torch_model_dict, paddle_model_dict, "decoder.attention_rnn", "decoder.attention_rnn")
    convert_linear(torch_model_dict, paddle_model_dict, "decoder.attention_layer.query_layer", "decoder.attention_layer.query_layer", bias=False)
    convert_linear(torch_model_dict, paddle_model_dict, "decoder.attention_layer.memory_layer", "decoder.attention_layer.memory_layer", bias=False)
    convert_linear(torch_model_dict, paddle_model_dict, "decoder.attention_layer.v", "decoder.attention_layer.value", bias=False)
    convert_conv1d(torch_model_dict, paddle_model_dict, "decoder.attention_layer.location_layer.location_conv.conv", "decoder.attention_layer.location_layer.location_conv", bias=False)
    convert_linear(torch_model_dict, paddle_model_dict, "decoder.attention_layer.location_layer.location_dense", "decoder.attention_layer.location_layer.location_dense", bias=False)
    convert_lstm(torch_model_dict, paddle_model_dict, "decoder.decoder_rnn", "decoder.decoder_rnn", reverse=False)
    convert_linear(torch_model_dict, paddle_model_dict, "decoder.linear_projection", "decoder.linear_projection", bias=True)
    convert_linear(torch_model_dict, paddle_model_dict, "decoder.gate_layer", "decoder.gate_layer", bias=True)
    for i in range(5):
        convert_conv1d(torch_model_dict, paddle_model_dict, "postnet.convolutions.{}.0.conv".format(i), "postnet.conv_list_{}".format(i))
        convert_batchnorm(torch_model_dict, paddle_model_dict, "postnet.convolutions.{}.1".format(i), "postnet.batch_norm_list_{}".format(i))

def build_paddle_model_dict(torch_model_dict):
    paddle_model_dict = OrderedDict()
    convert_tacotron2_encoder(torch_model_dict, paddle_model_dict)
    convert_tacotron2_decoder(torch_model_dict, paddle_model_dict)
    return paddle_model_dict

if __name__ == "__main__":
    checkpoint = torch.load('/paddle/tacotron2/outdir_test/checkpoint_0', map_location=torch.device('cpu'))
    #checkpoint.keys() == dict_keys(['optimizer', 'model'])
    torch_model_dict = checkpoint['state_dict']

    '''
    num = 0
    for key in torch_model_dict.keys():
        num += 1
        print('{}: shape:{}'.format(key, torch_model_dict[key].shape))
    print('total params:', num)
    exit()
    '''
    paddle_model_dict =  build_paddle_model_dict(torch_model_dict)
    file_name = 'experiment/checkpoints/paddle_param.pdparams'
    with open(file_name, 'wb') as f:
        pickle.dump(paddle_model_dict, f, protocol=2)
    print("Finish!")