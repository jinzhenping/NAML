import os
import logging
import fnmatch
import random

import numpy as np
import utils


def get_files(dirname, filename_pat="*", recursive=False):
    if not os.path.isdir(dirname):
        return None
    files = []
    for x in os.listdir(dirname):
        path = os.path.join(dirname, x)
        if os.path.isdir(path):
            if recursive:
                files.extend(get_files(path, filename_pat))
        elif fnmatch.fnmatch(x, filename_pat):
            files.append(path)
    return files


def get_worker_files(dirname,
                     worker_rank,
                     world_size,
                     filename_pat="*",
                     shuffle=False,
                     seed=0):
    all_files = get_files(dirname, filename_pat)
    if not all_files:
        raise FileNotFoundError(
            f"behavior files not found: dir={dirname} pat={filename_pat}"
        )
    all_files.sort()
    if shuffle:
        random.seed(seed)
        random.shuffle(all_files)
    files = []
    for i in range(worker_rank, len(all_files), world_size):
        files.append(all_files[i])
    logging.info(
        f"worker_rank:{worker_rank}, world_size:{world_size}, shuffle:{shuffle}, seed:{seed}, directory:{dirname}, files:{files}"
    )
    return files


class StreamReader:
    def __init__(self, data_paths, batch_size, shuffle=False, shuffle_buffer_size=1000):
        self.data_paths = list(data_paths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.lines = []
        self.idx = 0
        self.endofstream = False

    def reset(self):
        lines = []
        for p in self.data_paths:
            with open(p, "rb") as f:
                for line in f:
                    line = line.rstrip(b"\n").rstrip(b"\r")
                    if line:
                        lines.append(line)
        if self.shuffle:
            random.shuffle(lines)
        self.lines = lines
        self.idx = 0
        self.endofstream = False

    def get_next(self):
        if self.idx >= len(self.lines):
            self.endofstream = True
            return None
        batch = self.lines[self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size
        if self.idx >= len(self.lines):
            self.endofstream = True
        return np.array(batch, dtype=object)

    def reach_end(self):
        return self.endofstream


class StreamSampler:
    def __init__(
        self,
        data_dir,
        filename_pat,
        batch_size,
        worker_rank,
        world_size,
        enable_shuffle=False,
        shuffle_buffer_size=800,
        shuffle_seed=0,
    ):
        data_paths = get_worker_files(
            data_dir,
            worker_rank,
            world_size,
            filename_pat,
            shuffle=enable_shuffle,
            seed=shuffle_seed,
        )
        self.stream_reader = StreamReader(
            data_paths,
            batch_size,
            enable_shuffle,
            shuffle_buffer_size,
        )

    def __iter__(self):
        self.stream_reader.reset()
        return self

    def __next__(self):
        next_batch = self.stream_reader.get_next()
        if next_batch is None:
            raise StopIteration
        return next_batch

    def reach_end(self):
        return self.stream_reader.reach_end()


class StreamReaderTest(StreamReader):
    pass


class StreamSamplerTest(StreamSampler):
    def __init__(
        self,
        data_dir,
        filename_pat,
        batch_size,
        worker_rank,
        world_size,
        enable_shuffle=False,
        shuffle_buffer_size=1000,
        shuffle_seed=0,
    ):
        data_paths = get_worker_files(
            data_dir,
            worker_rank,
            world_size,
            filename_pat,
            shuffle=enable_shuffle,
            seed=shuffle_seed,
        )
        self.stream_reader = StreamReaderTest(
            data_paths,
            batch_size,
            enable_shuffle,
            shuffle_buffer_size,
        )


if __name__ == "__main__":
    print("start")
    print(os.getcwd())
