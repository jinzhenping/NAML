#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MM-Rec CLIP 학습용 데이터로더 (글로벌 news index, ROI/BERT 불필요)."""
from __future__ import annotations

import logging
import random
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import torch
from torch.utils.data import IterableDataset

from streaming import StreamSampler
import utils


def _as_text(line):
    if isinstance(line, (bytes, bytearray, np.bytes_)):
        return line.decode("utf-8", errors="replace")
    return str(line)


class DataLoaderTrainClip(IterableDataset):
    def __init__(
        self,
        data_dir,
        filename_pat,
        args,
        world_size,
        worker_rank,
        cuda_device_idx,
        news_index,
        enable_prefetch=True,
        enable_shuffle=False,
        enable_gpu=True,
    ):
        self.data_dir = data_dir
        self.filename_pat = filename_pat
        self.npratio = args.npratio
        self.user_log_length = args.user_log_length
        self.batch_size = args.batch_size
        self.worker_rank = worker_rank
        self.world_size = world_size
        self.cuda_device_idx = cuda_device_idx
        self.sampler = None
        self.shuffle_buffer_size = args.shuffle_buffer_size
        self.enable_prefetch = enable_prefetch
        self.enable_shuffle = enable_shuffle
        self.enable_gpu = enable_gpu
        self.epoch = -1
        self.news_index = news_index
        self.aval_count = 0
        self.stopped = False
        self.outputs = None
        self.pool = None

    def start(self):
        self.epoch += 1
        self.sampler = StreamSampler(
            data_dir=self.data_dir,
            filename_pat=self.filename_pat,
            batch_size=self.batch_size,
            worker_rank=self.worker_rank,
            world_size=self.world_size,
            enable_shuffle=self.enable_shuffle,
            shuffle_buffer_size=self.shuffle_buffer_size,
            shuffle_seed=self.epoch,
        )
        self.sampler.__iter__()

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[:fix_length] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, mask

    def newsample(self, news, ratio):
        if ratio > len(news):
            return news + [0] * (ratio - len(news))
        return random.sample(news, ratio)

    def start_async(self):
        self.aval_count = 0
        self.stopped = False
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def _produce(self):
        if self.enable_gpu:
            torch.cuda.set_device(self.cuda_device_idx)
        try:
            self.epoch += 1
            self.sampler = StreamSampler(
                data_dir=self.data_dir,
                filename_pat=self.filename_pat,
                batch_size=self.batch_size,
                worker_rank=self.worker_rank,
                world_size=self.world_size,
                enable_shuffle=self.enable_shuffle,
                shuffle_buffer_size=self.shuffle_buffer_size,
                shuffle_seed=self.epoch,
            )
            for batch in self.sampler:
                if self.stopped:
                    break
                context = self._process(batch)
                if context is None:
                    continue
                self.outputs.put(context)
                self.aval_count += 1
            self.outputs.put(None)
            self.aval_count += 1
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise

    def _process(self, batch):
        user_feature_batch, log_mask_batch, news_feature_batch, label_batch = [], [], [], []
        for line in batch:
            line = _as_text(line)
            splited = line.replace("\n", "").split("\t")
            if len(splited) < 8:
                continue
            history = [x for x in splited[3].split(" ") if x][-self.user_log_length :]
            poss = [x for x in splited[6].split(" ") if x]
            neg = [x for x in splited[7].split(" ") if x]
            if not poss:
                continue
            click_docs, log_mask = self.pad_to_fix_len(
                self.trans_to_nindex(history), self.user_log_length
            )
            for pdoc in poss:
                negps = self.newsample(neg, self.npratio)
                sample_news = self.trans_to_nindex([pdoc] + negps)
                user_feature_batch.append(click_docs)
                log_mask_batch.append(log_mask)
                news_feature_batch.append(sample_news)
                label_batch.append(0)
        if not user_feature_batch:
            return None
        if self.enable_gpu:
            return (
                torch.LongTensor(user_feature_batch).cuda(),
                torch.FloatTensor(log_mask_batch).cuda(),
                torch.LongTensor(news_feature_batch).cuda(),
                torch.LongTensor(label_batch).cuda(),
            )
        return (
            torch.LongTensor(user_feature_batch),
            torch.FloatTensor(log_mask_batch),
            torch.LongTensor(news_feature_batch),
            torch.LongTensor(label_batch),
        )

    def __iter__(self):
        logging.info("DataLoaderTrainClip __iter__()")
        if self.enable_prefetch:
            self.join()
            self.start_async()
        else:
            self.start()
        return self

    def __next__(self):
        while True:
            if self.enable_prefetch:
                if self.sampler and getattr(self.sampler, "reach_end", lambda: False)() and self.aval_count == 0:
                    raise StopIteration
                next_batch = self.outputs.get()
                self.outputs.task_done()
                self.aval_count -= 1
                if next_batch is None:
                    self.join()
                    raise StopIteration
            else:
                next_batch = self._process(self.sampler.__next__())
            if next_batch is None:
                continue
            return next_batch

    next = __next__

    def join(self):
        self.stopped = True
        if self.sampler and self.enable_prefetch and self.outputs is not None:
            while self.outputs.qsize() > 0:
                self.outputs.get()
                self.outputs.task_done()
            try:
                self.outputs.join()
            except Exception:
                pass
            if self.pool is not None:
                self.pool.shutdown(wait=True)
        self.sampler = None
