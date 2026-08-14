import os
import Model
import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F
import itertools
import yaml
import random
import argparse
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from datetime import datetime
# from Dataset.ma_seq import MASeq, Big_MASeq_Test_h5, Big_MASeq_Train_h5
from Dataset.ma_seq_merge import MASeq, Big_MASeq_Test_h5, Big_MASeq_Train_h5
from Utils.utils import load_train_data_BEAT, load_test_data_BEAT, visualize_and_write, load_train_info_BEAT_large, load_test_info_BEAT_large
from Utils.log import Logger
from easydict import EasyDict


class MoGPT:
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self.device = torch.device('cuda')
        self._build()

    def train(self):
        self.model_vqvae.eval()

        print("we use vqvae-model:", self.config.vqvae_weight)
        self.model_vqvae.load_state_dict(torch.load(self.config.vqvae_weight)['model'], strict=False)
        if hasattr(self.config, 'init_weight') and (self.config.init_weight is not None) and (
                self.config.init_weight != ''):
            print('Use pretrained model')
            print(self.config.init_weight)
            self.model_gpt.load_state_dict(torch.load(self.config.init_weight)['model'], strict=False)
        print(torch.load(self.config.vqvae_weight)['config'])

        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.config.cuda:
            torch.cuda.manual_seed(self.config.seed)

        for batch in self.train_loader:
            body_seqs, audio_feats, start_frame_num = batch
            print(body_seqs.shape, audio_feats.shape, start_frame_num)
            body_seqs = body_seqs.to(self.device)
            audio_feats = audio_feats.to(self.device)

            with torch.no_grad():
                quants_pred = self.model_vqvae.module.encode(body_seqs)

            if start_frame_num[0] < 10:
                print(start_frame_num, quants_pred)
            # if start_frame_num[0] > 2400:
            #     print(start_frame_num, quants_pred)


    def _build_model(self):
        print(f'Using {self.config.structure_vqvae.name} and {self.config.structure_gpt.name}')

        self.model_vqvae = torch.nn.parallel.DistributedDataParallel(
            getattr(Model, self.config.structure_vqvae.name)(self.config.structure_vqvae).cuda(),
            device_ids=[self.config.local_rank],
            output_device=self.config.local_rank,
            find_unused_parameters=True
        )
        self.model_gpt = torch.nn.parallel.DistributedDataParallel(
            getattr(Model, self.config.structure_gpt.name)(self.config.structure_gpt).cuda(),
            device_ids=[self.config.local_rank],
            output_device=self.config.local_rank,
            find_unused_parameters=True
        )

    def _build(self):
        self._build_model()
        if not (hasattr(self.config, 'need_not_train_data') and self.config.need_not_train_data):
            self._build_train_loader()

    def _build_train_loader(self):
        print("Build train loader")
        data = self.config.data
        if data.name == "BEAT" or data.name == "zeroeggs" or data.name == "mocap":
            print("Train with BEAT dataset")
            train_file_names, train_body_motions_len, train_audio_features_len = load_train_info_BEAT_large(
                data_dir=data.dir,
                test_files=data.test_files,
                window_size=data.seq_len_train,
                stride=data.stride_train
            )
        else:
            raise ValueError

        train_dataset = Big_MASeq_Train_h5(
            data_dir = data.dir,
            body_motions_name = train_file_names,
            body_motions_len = train_body_motions_len,
            window_size = data.seq_len_train,
            stride = data.stride_train,
            ds_rate = data.ds_rate,
            return_start_frame_num=True
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=1,
            # shuffle=True,
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler
        )

        self.train_loader = data_loader
        self.train_sampler = train_sampler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Pytorch implementation"
    )

    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--local-rank', default=-1, type=int,
                        help='node rank for distributed training')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--eval', action='store_true')

    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    with open(args.config, "r") as f:
        config = EasyDict(yaml.safe_load(f))
    config['local_rank'] = args.local_rank

    agent = MoGPT(config)
    agent.train()
