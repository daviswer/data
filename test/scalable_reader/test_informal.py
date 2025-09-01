import functools
from copy import deepcopy
import math
import os
import pyarrow as pa
import shutil
import tempfile
import torch

from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.scalable_reader import (
    ScalableReader,
    PreprocessDataset,
    SamplingDataset,
    ShuffleDataset,
    DocPackingDataset,
    ArrowHandler,
    save_ckpt_custom,
    load_ckpt_custom,
    _StatefulDataset,
)

# Generates test data in a temp directory, and returns that tempdir object.
# (file path can be retrieved via tempdir.name)
# Two dataset folders: one has a large shardfile (100x100), other has two small shardfiles (50x50)
def generate_sequential_multidata():
    tmpdir = tempfile.TemporaryDirectory()
    schema = pa.schema([pa.field("tokens", pa.uint32())])

    os.mkdir(os.path.join(tmpdir.name, "dataset_1"))
    os.mkdir(os.path.join(tmpdir.name, "dataset_2"))
    os.mkdir(os.path.join(tmpdir.name, "dataset_2", "subfolder"))
    with pa.ipc.new_file(
        os.path.join(tmpdir.name, "dataset_1/fullshard.arrow"), schema
    ) as writer:
        for i in range(100):
            out = list(range(i * 100, i * 100 + 99))  # 99 because added delimiter makes it 100
            writer.write(pa.record_batch([out], schema=schema))

    with pa.ipc.new_file(
        os.path.join(tmpdir.name, "dataset_2/quartershard_1.arrow"), schema
    ) as writer:
        for i in range(50):
            out = list(range(i * 50, i * 50 + 49))  # 49 because added delimiter makes it 50
            writer.write(pa.record_batch([out], schema=schema))

    with pa.ipc.new_file(
        os.path.join(tmpdir.name, "dataset_2/subfolder/quartershard_2.arrow"), schema
    ) as writer:
        for i in range(50):
            out = list(range(2500 + i * 50, 2500 + i * 50 + 49))  # 49 because added delimiter makes it 50
            writer.write(pa.record_batch([out], schema=schema))

    return tmpdir

# Make mock data for re-use. Returns directory path.
tmpdir = generate_sequential_multidata()
path = tmpdir.name

# Create a save directory for testing checkpoints
ckptpath = os.path.join(path, "ckpts")
os.makedirs(ckptpath)

def pipeline(
    # Base reader
    path = path,
    rank = 0,
    worldsize = 1,
    delimiter = -1,
    seed = 42,
    chunk = 1000,
    logicals = 10,
    # Sampling
    sample = True,
    datasets = ["dataset_1","dataset_2"],
    weights = [2, 1],
    # Packing
    pack = True,
    seqlen = 100,
    npads = 0,
    pad = -2,
    nbins = 1,
    # Shuffling
    shuffle = True,
    window = 10,
    # Tensor
    tensor = True,
):
    # Build dataloader
    data = ScalableReader(
        path, 
        rank, 
        worldsize, 
        ArrowHandler, 
        delimiter, 
        None, 
        seed=seed, 
        max_chunksize=chunk, 
        n_logical_shards=logicals,
    )
    if sample:
        data = SamplingDataset(path, data, delimiter, datasets, weights)
    if pack:
        data = DocPackingDataset(data, seqlen, npads, delimiter, pad, nbins)
    if shuffle:
        data = ShuffleDataset(data, window)
    if tensor:
        # Statelessly convert all outputs to tensors
        data = PreprocessDataset(data, torch.tensor)
    return data

basicdata = functools.partial(pipeline, sample=False, pack=False, shuffle=False)
basicdata1 = functools.partial(basicdata, path = os.path.join(path, "dataset_1"))
basicdata2 = functools.partial(basicdata, path = os.path.join(path, "dataset_2"))
basicmessy = functools.partial(
    pipeline, 
    worldsize=17, 
    chunk=27,
    logicals=37,
    weights=[7,3],
    seqlen=37,
    npads=3,
    nbins=17,
    window=27,
)

def test_single_epoch():
    # For varying worldsizes, logical shard partitions, and chunk sizes,
    # ensure every data point is viewed at least once per epoch.
    # Also ensure that independent ranks do not share data.
    size_set = (1,10,37,100,137)
    for n_logical in size_set:
        for worldsize in size_set:
            for chunksize in (100,50):
                if worldsize <= n_logical and worldsize < 100:
                    out = []
                    max_docs_per_shard = math.ceil(100/worldsize)
                    max_shards_per_worker = math.ceil(n_logical/worldsize)
                    nsteps = max_shards_per_worker * max_docs_per_shard * (100//chunksize)
                    for r in range(worldsize):
                        out.append(set())
                        data = basicdata1(logicals=n_logical, worldsize=worldsize, rank=r, chunk=chunksize)
                        loader = iter(data)
                        for _ in range(nsteps):
                            out[-1].add(next(loader)[0].item())
                    
                    # Check that all data is accounted for
                    all_out = set().union(*out)
                    expected = set([x*chunksize for x in range(100 * (100//chunksize))])
                    missing = expected.difference(expected.intersection(all_out))
                    assert len(missing) == 0, f"Following data points missing from epoch (logicals {n_logical}, worldsize {worldsize}, {nsteps} steps): {missing}"

                    # Check that ranks do not overlap
                    for i,r1 in enumerate(out):
                        for j,r2 in enumerate(out):
                            if i<j:
                                overlap = sorted(list(r1.intersection(r2)))
                                assert len(overlap) == 0, f"Ranks {i} and {j} (logicals {n_logical}, worldsize {worldsize}) contain overlap {overlap}"


def test_packing_epoch():
    # For varying bucket/seqlen/pad counts, ensure every data point is viewed at least once per epoch.

    def first_tokens(x):
        # Pull out all tokens in a packed sequence that could be starting a doc
        delimiters = [i for i,t in enumerate(x) if t==-1 and i < len(x)-1]
        return [x[0]] + [x[i+1] for i in delimiters]
        
    for buckets in (1,2,4,8,16):
        for pad in (0,5,10):
            for seqlen in (17,77,117):
                data = basicdata1(chunk=30, pack=True, seqlen=seqlen, npads=pad, nbins=buckets, tensor=False)
                loader = iter(data)
                nsteps = math.ceil(100*100 / (seqlen - pad))
                out = set()
                epoch = -1
                for _ in range(nsteps):
                    x = next(loader)
                    out.update(set(first_tokens(x)))
                    if 0 in x:
                        epoch += 1
                        if epoch == 1:
                            break
                
                # Add any unpassed buckets
                for seq in data.bins:
                    out.update(set(first_tokens(seq)))
                
                # Check that all data is accounted for
                expected = set([x*100 for x in range(100)])
                missing = expected.difference(expected.intersection(out))
                assert len(missing) == 0, f"Following data points missing from epoch (buckets {buckets}, pad {pad}, seqlen {seqlen}): {missing}"


def test_sampler_ratios():
    # Single worker, varying weights: verify that loaders pull subdatasets at regular intervals
    # (when data and doc sizes are regular and divisible). Verify that most-undersampled is being selected.
    weights = [[1, 1], [2, 1], [2, 3], [2, 5]]
    target_rate = [3, 2, 4, 6]
    
    # Expected sequences for each case are:
    # 1 2 2 (1 2 2)...
    # 1 2 (1 2)...
    # 1 2 2 2 (1 2 2 2)...
    # 1 2 2 2 2 2 (1 2 2 2 2 2)...

    def check_rates(w, t):
        s = []
        d = basicdata(sample=True, datasets=["dataset_1", "dataset_2"], weights=w)
        l = iter(d)
        for i in range(100):
            out = next(l)
            s.append(len(out))
            if i % t == 0:
                assert (
                    len(out) == 100
                ), f"Output {i} length {len(out)} does not match expected 100. Sequence so far: {s}, round {t}"
            else:
                assert (
                    len(out) == 50
                ), f"Output {i} length {len(out)} does not match expected 50. Sequence so far: {s}, round {t}"

    for i in range(4):
        check_rates(weights[i], target_rate[i])


def test_packing_nobucket():
    # Wrap a simple generator that spits out varying-length incremental counts. Wrap it in a packer/slicer
    # with only one bucket, and verify across seeds / sequence lengths that data is packed appropriately.

    class RandCounter(_StatefulDataset):
        # Spit out incremental counts of random length, uniformly sampled from 2 to maxlen
        def __init__(self, seed, maxlen):
            self.i = 0
            self.rank = 0
            self.worldsize = 1
            self.datapath = tmpdir.name
            self.g = torch.Generator().manual_seed(seed)
            self.is_setup = True
            self.maxlen = maxlen

        def __iter__(self):
            while True:
                l = torch.randint(2, self.maxlen, [1], generator=self.g).item()
                out = list(range(self.i, self.i + l))
                out[-1] = -1
                yield out
                self.i += l

    for seed in [0,1,7,42,777,2025]:
        for seqlen in [32, 50, 77, 100]:
            data = DocPackingDataset(RandCounter(seed, seqlen), seqlen, 0, -1, -2, 1)
            loader = iter(data)
            for i in range(100):
                out = next(loader)
                if out[-1] != -1:
                    assert out[-1] == (i+1)*seqlen - 1, f"Output does not match, Got {out[-5:]}, expected {(i+1)*seqlen-1}"
                else:
                    assert out[-2] == (i+1)*seqlen - 2, f"Output does not match, Got {out[-5:]}, expected {(i+1)*seqlen-2}"
                return


def test_shuffle_coverage():
    # Wrap a simple generator that spits out constant incremental sequences. Wrap it in a shuffler,
    # and verify across seeds, steps and buffer sizes that n% of the first (100-n)% of data points 
    # are emitted.

    class SteadyCounter(_StatefulDataset):
        # Spit out incremental counts of random length, uniformly sampled from 2 to maxlen
        def __init__(self, seed, maxlen):
            self.i = 0
            self.rank = seed
            self.worldsize = seed+1
            self.datapath = tmpdir.name
            self.g = torch.Generator().manual_seed(seed)
            self.is_setup = True
            self.maxlen = maxlen

        def __iter__(self):
            while True:
                out = list(range(self.i, self.i + self.maxlen))
                out[-1] = -1
                yield out
                self.i += self.maxlen

    for seed in [0,1,7,42,777,2025]:
        for window in [100, 250, 777]:
            data = ShuffleDataset(SteadyCounter(seed, 10), window)
            loader = iter(data)
            out = set()
            for step in range(1000):
                x = next(loader)[0]
                out.add(x)
                if (step+1)%200 == 0:
                    for n in [.75,.8,.85,.9,.95]:
                        expected = set([i*10 for i in range(math.ceil((step+1)*(1-n)))])
                        targ = math.floor((step+1)*n*(1-n))
                        overlap = expected.intersection(out)
                        assert len(overlap) >= targ, f"Values missing. Needed {targ} overlapping tokens, instead got {len(overlap)}: {overlap}"


def test_reload():
    # For varying dataloader pipelines and shard/worker counts, reload from checkpoint,
    # without rescaling, and ensure output continues to match
    shardcounts = [10,27,77]
    workercounts = [1,3,5]
    stepcounts = [0,17,37]
    pipelines = [
        basicdata1,
        functools.partial(basicdata1, pack=True, seqlen=37, nbins=10),
        functools.partial(basicdata1, shuffle=True, pack=True, seqlen=47, nbins=7, npads=5),
    ]
    for steps in stepcounts:
        for shards in shardcounts:
            for workers in workercounts:
                for pipe in pipelines:
                    datas = [pipe(rank=i, worldsize=workers, logicals=shards) for i in range(workers)]
                    loaders = [iter(d) for d in datas]
                    for _ in range(steps):
                        [next(l) for l in loaders]
                    d = [d.state_dict() for d in datas]

                    # Load into new pipeline and compare outputs
                    datas2 = [pipe(rank=i, worldsize=workers, logicals=shards) for i in range(workers)]
                    [datas2[i].load_state_dict(d[i]) for i in range(workers)]
                    loaders2 = [iter(d) for d in datas2]
                    for _ in range(20):
                        for i in range(workers):
                            out = next(loaders[i])
                            out2 = next(loaders2[i])
                            assert out.sub(out2).sum().item() == 0


def test_stateful_equal():
    # Ensure that list[pipeline] outputs are equivalent to StatefulDataLoader[pipeline], both before
    # and after reloading from checkpoint.
    loader = StatefulDataLoader(basicmessy(rank=0, worldsize=1), batch_size=1, num_workers=3)
    datas = [basicmessy(rank=i, worldsize=3) for i in range(3)]
    loaders = [iter(data) for data in datas]

    for i, out in enumerate(loader):
        out2 = set(next(loaders[i%3]).tolist())
        out = set(out.squeeze().tolist())
        assert len(out.intersection(out2)) == len(out), [i, out, out2]
        if i==29:
            break
    
    save_ckpt_custom(loader, os.path.join(path, "ckpt"))
    states = [d.state_dict() for d in datas]

    loader2 = StatefulDataLoader(basicmessy(rank=0, worldsize=1), batch_size=1, num_workers=3)
    datas2 = [basicmessy(rank=i, worldsize=3) for i in range(3)]
    load_ckpt_custom(loader2, os.path.join(path, "ckpt"))
    [datas2[i].load_state_dict(states[i]) for i in range(3)]
    loaders2 = [iter(data) for data in datas2]

    for i, out in enumerate(loader2):
        out2 = set(next(loaders2[i%3]).tolist())
        out = set(out.squeeze().tolist())
        assert len(out.intersection(out2)) == len(out), [i, out, out2]
        if i==30:
            break


def test_rescale_epoch():
    # Complete part of an epoch, then rescale. Verify that until epoch is complete, data does not
    # repeat, and that once epoch is complete, all data has appeared. Due to long setup time of
    # StatefulDataLoader, we forego the usual nested for-loops.
    devices1 = [1, 1, 2]
    devices2 = [1, 2, 1]
    workers1 = [1, 3, 3]
    workers2 = [2, 2, 5]
    logicals = [23, 37, 41]
    steps = [50, 15, 10]
    for d1,d2,w1,w2,l,nsteps in zip(devices1, devices2, workers1, workers2, logicals, steps):
        # Clear any prior ckpts
        ckptpath = os.path.join(path, "rescale_ckpt")
        if os.path.exists(ckptpath):
            shutil.rmtree(ckptpath)

        # Take first round of steps
        loaders = [StatefulDataLoader(
            basicdata1(rank=i, worldsize=d1, logicals=l), 
            num_workers=w1,
        ) for i in range(d1)]
        test = []
        for loader in loaders:
            for i,out in enumerate(loader):
                test.append(out.squeeze()[0].item())
                if i == w1*nsteps-1:
                    break
            save_ckpt_custom(loader, ckptpath)

        # Calculate the max and min number of steps to complete the epoch
        min_logical_size = math.floor(100/l)
        max_logical_size = math.ceil(100/l)
        min_logicals = math.floor(l/d2/w2)
        max_logicals = math.ceil(l/d2/w2)
        min_extra_steps = min_logical_size * min_logicals - math.ceil(len(test)/w2/d2)
        max_extra_steps = max_logical_size * max_logicals - math.floor(len(test)/w2/d2)
        print([d1,d2,w1,w2,l], min_extra_steps, max_extra_steps)

        # Create second round of loaders
        loaders2 = [StatefulDataLoader(
            basicdata1(rank=i, worldsize=d2, logicals=l), 
            num_workers=w2,
        ) for i in range(d2)]

        # Take min number of steps to complete epoch, check overlap
        testmin = []
        testmax = []
        for loader in loaders2:
            load_ckpt_custom(loader, ckptpath)
            for i,out in enumerate(loader):
                x = out.squeeze()[0].item()
                testmax.append(x)
                if i < w2*min_extra_steps:
                    testmin.append(x)
                if i == w2*max_extra_steps-1:
                    break
                
        test = set(test)
        testmin = set(testmin)
        testmax = set(testmax)
        # Check for worker overlap (assuming no epoch overlap)
        assert len(testmin) == w2*d2*min_extra_steps, (f"Pre-epoch output has the wrong number of tokens: expected {w2*d2*min_extra_steps}, got {len(testmin)}", sorted(list(testmin)))
        assert len(testmax) == w2*d2*max_extra_steps, (f"Post-epoch output has the wrong number of tokens: expected {w2*d2*max_extra_steps}, got {len(testmax)}", sorted(list(testmax)))
        # Check for repetition of data before end of epoch
        assert len(testmin.intersection(test)) == 0, ("Overlap detected between first and second rounds:", test, testmin)
        # Check that all data is represented after full epoch
        testfull = test.union(testmax)
        basis = set(i*100 for i in range(100))
        assert len(testfull.intersection(basis)) == 100, ("Data missing from epoch:", sorted(list(testfull)))
        