import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_nsynth.nsynth import NSynth
import numpy as np
from matplotlib.pyplot import specgram


def get_data_loader(type, network="", **kwargs):
    assert (type in ("train", "test", "valid", )), "Invalid data loader type: {}".format(type)

    data_path = {
        "train": "/local/sandbox/nsynth/nsynth-train",
        "test" : "/local/sandbox/nsynth/nsynth-test",
        "valid": "/local/sandbox/nsynth/nsynth-valid"
    }

    categorical_field_list = ["instrument_family"]

    if network == "Bonus":
        categorical_field_list = ["instrument_source"]
        transformations = transforms.Compose([
            transforms.Lambda(lambda x: mfcc_to_tensor(x)), # Convert to 2d MFCC image
            transforms.Lambda(lambda x: x + 1), # Avoid underflow (NaN)
            transforms.Lambda(lambda x: np.expand_dims(x, axis=0)),
            transforms.Lambda(lambda x: F.interpolate(torch.tensor(x), scale_factor=0.25)) # Downscale
        ])

    else:
        transformations = transforms.Compose([
            transforms.Lambda(lambda x: (x / np.iinfo(np.int16).max) + 1), # Normalize to [0,2]
            transforms.Lambda(lambda x: x[:32000]),
            transforms.Lambda(lambda x: np.expand_dims(x, axis=0))
        ])


    dataset = NSynth(
        data_path[type],
        transform=transformations,
        blacklist_pattern=["synth_lead"],  # blacklist synth_lead instrument
        categorical_field_list=categorical_field_list)
    loader = data.DataLoader(dataset, **kwargs)
    return loader


def mfcc_to_tensor(samples, sr=16000):
    f = samples.flatten()
    spectrogram, _, _, im = specgram(f, Fs=sr)
    # spectrogram = torch.tensor(s)
    # img = im.get_array()
    # tensor = torch.tensor(np.flip(img, axis=0))
    return spectrogram