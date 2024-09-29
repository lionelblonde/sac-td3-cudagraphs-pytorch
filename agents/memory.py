from typing import Union

from beartype import beartype
from einops import repeat, pack, rearrange
import numpy as np
import torch


class RingBuffer(object):

    @beartype
    def __init__(self, maxlen: int, shape: tuple[int, ...], device: torch.device):
        """Ring buffer impl"""
        self.maxlen = maxlen
        self.device = device
        self.start = 0
        self.length = 0
        self.data = torch.zeros((maxlen, *shape), dtype=torch.float32, device=self.device)

    @beartype
    def __len__(self):
        return self.length

    @beartype
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.length:
            raise KeyError
        return self.data[(self.start + idx) % self.maxlen]

    @beartype
    def get_batch(self, idxs: torch.Tensor) -> torch.Tensor:
        # important: idxs is a tensor, and start and maxlen are ints
        return self.data[(self.start + idxs) % self.maxlen]

    @beartype
    def append(self, *, v: torch.Tensor):
        if self.length < self.maxlen:
            # we have space, simply increase the length
            self.length += 1
            self.data[(self.start + self.length - 1) % self.maxlen] = v
        elif self.length == self.maxlen:
            # no space, remove the first item
            self.start = (self.start + 1) % self.maxlen
            self.data[(self.start + self.length - 1) % self.maxlen] = v
        else:
            # this should never happen
            raise RuntimeError

    @property
    @beartype
    def latest_entry_idx(self) -> int:
        return (self.start + self.length - 1) % self.maxlen

    @beartype
    @classmethod
    def sanity_check_ringbuffer(cls):
        # create a RingBuffer object
        ring_buffer = cls((maxlen := 5), (shape := (3, 2)), device=torch.device("cpu"))
        # fill the replay buffer with maxlen+1 items
        new_items = [
            torch.rand(*shape),
            (i1 := torch.rand(*shape)),
            (i2 := torch.rand(*shape)),
            (i3 := torch.rand(*shape)),
            (i4 := torch.rand(*shape)),
            (i5 := torch.rand(*shape)),
        ]
        assert len(new_items) == maxlen + 1  # for us
        for i in range(maxlen + 1):
            ring_buffer.append(v=new_items[i])
        # check that the last item added is in first position
        assert torch.equal(ring_buffer[0], i1)
        assert torch.equal(ring_buffer[1], i2)
        assert torch.equal(ring_buffer[2], i3)
        assert torch.equal(ring_buffer[3], i4)
        assert torch.equal(ring_buffer[4], i5)


class ReplayBuffer(object):

    @beartype
    def __init__(self,
                 generator: torch.Generator,
                 capacity: int,
                 erb_shapes: dict[str, tuple[int, ...]],
                 device: torch.device):
        """Replay buffer impl"""
        self.rng = generator
        self.capacity = capacity
        self.erb_shapes = erb_shapes
        self.device = device
        self.ring_buffers = {
            k: RingBuffer(self.capacity, s, self.device) for k, s in self.erb_shapes.items()}

    @beartype
    def get_trns(self, idxs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Collect a batch from indices"""
        trns = {}
        for k, v in self.ring_buffers.items():
            trns[k] = v.get_batch(idxs)
        return trns

    @beartype
    def discount(self, x: torch.Tensor, gamma: float) -> torch.Tensor:
        """Compute gamma-discounted sum"""
        c = x.size(0)
        reps = repeat(x, "k 1 -> c k", c=c)  # note: k in einstein notation is c
        mats = [
            (gamma ** (c - j)) *
                torch.diagflat(torch.ones(j, device=self.device), offset=(c - j))
            for j in reversed(range(1, c + 1))]
        mats, _ = pack(mats, "* h w")
        out = rearrange(torch.sum(reps * torch.sum(mats, dim=0), dim=1), "k -> k 1")
        assert out.size() == x.size()
        return out[0]  # would be simpler to just compute the 1st elt, but only used in n-step rets

    @beartype
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample transitions uniformly from the replay buffer"""
        idxs = torch.randint(
            low=0,
            high=self.num_entries,
            size=(batch_size,),
            generator=self.rng,
            device=self.device,
        )
        return self.get_trns(idxs)

    @beartype
    def append(self, trn: dict[str, Union[np.ndarray, torch.Tensor]]):
        """Add a transition to the replay buffer."""
        assert set(self.ring_buffers.keys()) == set(trn.keys()), "key mismatch"
        for k in self.ring_buffers:
            if not isinstance(trn[k], (np.ndarray, torch.Tensor)):
                raise TypeError(k)
            if isinstance(trn[k], np.ndarray):
                new_tensor = torch.as_tensor(trn[k], device=self.device, dtype=torch.float)
            else:
                new_tensor = trn[k]
            assert isinstance(new_tensor, torch.Tensor)
            self.ring_buffers[k].append(v=new_tensor)

    @property
    @beartype
    def latest_entry(self) -> dict[str, torch.Tensor]:
        return self.get_trns(torch.tensor(self.latest_entry_idx))

    @property
    @beartype
    def latest_entry_idx(self) -> int:
        return self.ring_buffers["obs0"].latest_entry_idx  # could pick any other key

    @property
    @beartype
    def num_entries(self) -> int:
        return len(self.ring_buffers["obs0"])  # could pick any other key

    @property
    @beartype
    def how_filled(self) -> str:
        num = f"{self.num_entries:,}".rjust(10)
        denomi = f"{self.capacity:,}".rjust(10)
        return f"{num} / {denomi}"

    @beartype
    def __repr__(self) -> str:
        shapes = "|".join([f"[{k}:{s}]" for k, s in self.erb_shapes.items()])
        return f"ReplayBuffer(capacity={self.capacity}, shapes={shapes})"
