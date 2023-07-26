import torch
import torch.nn as nn
import numpy as np
import math
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from wav2vec import Wav2Vec2Model, Wav2Vec2ForSpeechClassification
from utils import init_biased_mask, enc_dec_mask


class EmoTalk(nn.Module):
    def __init__(self, args):
        super(EmoTalk, self).__init__()
        self.feature_dim = args.feature_dim
        self.bs_dim = args.bs_dim
        self.device = args.device
        self.batch_size = args.batch_size
        self.audio_encoder_cont = Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.audio_encoder_cont.feature_extractor._freeze_parameters()
        self.audio_encoder_emo = Wav2Vec2ForSpeechClassification.from_pretrained(
            "r-f/wav2vec-english-speech-emotion-recognition")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "r-f/wav2vec-english-speech-emotion-recognition")
        self.audio_encoder_emo.wav2vec2.feature_extractor._freeze_parameters()
        self.max_seq_len = args.max_seq_len
        self.audio_feature_map_cont = nn.Linear(1024, 512)
        self.audio_feature_map_emo = nn.Linear(1024, 832)
        self.audio_feature_map_emo2 = nn.Linear(832, 256)
        self.relu = nn.ReLU()
        self.biased_mask1 = init_biased_mask(n_head=4, max_seq_len=args.max_seq_len, period=args.period)
        self.one_hot_level = np.eye(2)
        self.obj_vector_level = nn.Linear(2, 32)
        self.one_hot_person = np.eye(24)
        self.obj_vector_person = nn.Linear(24, 32)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=args.feature_dim,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.bs_map_r = nn.Linear(self.feature_dim, self.bs_dim)
        nn.init.constant_(self.bs_map_r.weight, 0)
        nn.init.constant_(self.bs_map_r.bias, 0)

    def forward(self, data):
        frame_num11 = data["target11"].shape[1]
        frame_num12 = data["target12"].shape[1]
        inputs12 = self.processor(torch.squeeze(data["input12"]), sampling_rate=16000, return_tensors="pt",
                                  padding="longest").input_values.to(self.device)
        hidden_states_cont1 = self.audio_encoder_cont(inputs12, frame_num=frame_num11).last_hidden_state
        hidden_states_cont12 = self.audio_encoder_cont(inputs12, frame_num=frame_num12).last_hidden_state
        inputs21 = self.feature_extractor(torch.squeeze(data["input21"]), sampling_rate=16000, padding=True,
                                          return_tensors="pt").input_values.to(self.device)
        inputs12 = self.feature_extractor(torch.squeeze(data["input12"]), sampling_rate=16000, padding=True,
                                          return_tensors="pt").input_values.to(self.device)

        output_emo1 = self.audio_encoder_emo(inputs21, frame_num=frame_num11)
        output_emo2 = self.audio_encoder_emo(inputs12, frame_num=frame_num12)

        hidden_states_emo1 = output_emo1.hidden_states
        hidden_states_emo2 = output_emo2.hidden_states

        label1 = output_emo1.logits
        onehot_level = self.one_hot_level[data["level"]]
        onehot_level = torch.from_numpy(onehot_level).to(self.device).float()
        onehot_person = self.one_hot_person[data["person"]]
        onehot_person = torch.from_numpy(onehot_person).to(self.device).float()
        if data["target11"].shape[0] == 1:
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0)
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0)
        else:
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0).permute(1, 0, 2)
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0).permute(1, 0, 2)

        obj_embedding_level11 = obj_embedding_level.repeat(1, frame_num11, 1)
        obj_embedding_level12 = obj_embedding_level.repeat(1, frame_num12, 1)
        obj_embedding_person11 = obj_embedding_person.repeat(1, frame_num11, 1)
        obj_embedding_person12 = obj_embedding_person.repeat(1, frame_num12, 1)
        hidden_states_cont1 = self.audio_feature_map_cont(hidden_states_cont1)
        hidden_states_emo11_832 = self.audio_feature_map_emo(hidden_states_emo1)
        hidden_states_emo11_256 = self.relu(self.audio_feature_map_emo2(hidden_states_emo11_832))

        hidden_states11 = torch.cat(
            [hidden_states_cont1, hidden_states_emo11_256, obj_embedding_level11, obj_embedding_person11], dim=2)
        hidden_states_cont12 = self.audio_feature_map_cont(hidden_states_cont12)
        hidden_states_emo12_832 = self.audio_feature_map_emo(hidden_states_emo2)
        hidden_states_emo12_256 = self.relu(self.audio_feature_map_emo2(hidden_states_emo12_832))

        hidden_states12 = torch.cat(
            [hidden_states_cont12, hidden_states_emo12_256, obj_embedding_level12, obj_embedding_person12], dim=2)
        if data["target11"].shape[0] == 1:
            tgt_mask11 = self.biased_mask1[:, :hidden_states11.shape[1], :hidden_states11.shape[1]].clone().detach().to(
                device=self.device)
            tgt_mask22 = self.biased_mask1[:, :hidden_states12.shape[1], :hidden_states12.shape[1]].clone().detach().to(
                device=self.device)

        memory_mask11 = enc_dec_mask(self.device, hidden_states11.shape[1], hidden_states11.shape[1])
        memory_mask12 = enc_dec_mask(self.device, hidden_states12.shape[1], hidden_states12.shape[1])
        bs_out11 = self.transformer_decoder(hidden_states11, hidden_states_emo11_832, tgt_mask=tgt_mask11,
                                            memory_mask=memory_mask11)
        bs_out12 = self.transformer_decoder(hidden_states12, hidden_states_emo12_832, tgt_mask=tgt_mask22,
                                            memory_mask=memory_mask12)
        bs_output11 = self.bs_map_r(bs_out11)
        bs_output12 = self.bs_map_r(bs_out12)

        return bs_output11, bs_output12, label1

    def predict(self, audio, level, person):
        frame_num11 = math.ceil(audio.shape[1] / 16000 * 30)
        inputs12 = self.processor(torch.squeeze(audio), sampling_rate=16000, return_tensors="pt",
                                  padding="longest").input_values.to(self.device)
        hidden_states_cont1 = self.audio_encoder_cont(inputs12, frame_num=frame_num11).last_hidden_state
        inputs12 = self.feature_extractor(torch.squeeze(audio), sampling_rate=16000, padding=True,
                                          return_tensors="pt").input_values.to(self.device)
        output_emo1 = self.audio_encoder_emo(inputs12, frame_num=frame_num11)
        hidden_states_emo1 = output_emo1.hidden_states

        onehot_level = self.one_hot_level[level]
        onehot_level = torch.from_numpy(onehot_level).to(self.device).float()
        onehot_person = self.one_hot_person[person]
        onehot_person = torch.from_numpy(onehot_person).to(self.device).float()
        if audio.shape[0] == 1:
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0)
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0)
        else:
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0).permute(1, 0, 2)
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0).permute(1, 0, 2)

        obj_embedding_level11 = obj_embedding_level.repeat(1, frame_num11, 1)
        obj_embedding_person11 = obj_embedding_person.repeat(1, frame_num11, 1)
        hidden_states_cont1 = self.audio_feature_map_cont(hidden_states_cont1)
        hidden_states_emo11_832 = self.audio_feature_map_emo(hidden_states_emo1)
        hidden_states_emo11_256 = self.relu(
            self.audio_feature_map_emo2(hidden_states_emo11_832))

        hidden_states11 = torch.cat(
            [hidden_states_cont1, hidden_states_emo11_256, obj_embedding_level11, obj_embedding_person11], dim=2)
        if audio.shape[0] == 1:
            tgt_mask11 = self.biased_mask1[:, :hidden_states11.shape[1],
                         :hidden_states11.shape[1]].clone().detach().to(device=self.device)

        memory_mask11 = enc_dec_mask(self.device, hidden_states11.shape[1], hidden_states11.shape[1])
        bs_out11 = self.transformer_decoder(hidden_states11, hidden_states_emo11_832, tgt_mask=tgt_mask11,
                                            memory_mask=memory_mask11)
        bs_output11 = self.bs_map_r(bs_out11)

        return bs_output11
