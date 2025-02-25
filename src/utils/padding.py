import torch
import numpy as np


def pad_mel_spectrogram(mel, max_len):
    pad_amount = max_len - mel.shape[1]
    return torch.nn.functional.pad(mel, (0, pad_amount)) 

def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
       
        max_len = max(np.shape(x)[1] for x in inputs)
        output = [pad_mel_spectrogram(torch.tensor(mel), torch.tensor(max_len)) for mel in inputs] 

    return output
    
def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded