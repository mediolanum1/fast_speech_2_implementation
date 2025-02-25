import torch
from src.utils.padding import pad_1D, pad_2D, pad_mel_spectrogram

def reprocess( data, idxs):

    audio = [data[idx]["audio"] for idx in idxs]
    audio_paths = [data[idx]["audio_path"] for idx in idxs]
    #phonemes_encoded = [data[idx]["phonemes_encoded"] for idx in idxs]
    texts = [data[idx]["text_encoded"] for idx in idxs]
    raw_texts = [data[idx]["text"] for idx in idxs]
    mels = [data[idx]["mel"] for idx in idxs]
    pitches = [data[idx]["pitch"] for idx in idxs]
    energies = [data[idx]["energy"] for idx in idxs]
    audio_len = [data[idx]["audio_len"] for idx in idxs]
    duration = [data[idx]["durations"] for idx in idxs]
    text_lens = np.array([text.shape[0] for text in texts])
    mel_lens = np.array([mel.shape[0] for mel in mels])
    audio_lens = [data[idx]["audio_len"] for idx in idxs]
    phonemes = [data[idx]["phonemes"] for idx in idxs]
    texts = pad_1D(texts)
    mels = pad_2D(mels)
    pitches = pad_1D(pitches)
    energies = pad_1D(energies)
#    phonemes_encoded = pad_1D(phonemes_encoded)
    durations = pad_1D(duration)
    
    return {
            "audios": audio,
            "audio_lens": audio_lens,
            "audio_paths": audio_paths,
            "texts": raw_texts,
            "text_encoded": texts,
            "text_lens": text_lens,
            "max_text_lens": max(text_lens),
          #  "phonemes_encoded": phonemes_encoded,
            "phonemes": phonemes,
            "durations": durations,
            "mels": mels,
            "mel_lens": mel_lens,
            "max_mel_lens": max(mel_lens),
            "pitch": pitches,
            "energy": energies }

def collate_fn(data: list[dict], sort: bool = True, drop_last:bool = False, batch_size: int = 40):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    data_size = len(data)

    if sort:
        len_arr = np.array([d["text_encoded"].shape[0] for d in data])
        idx_arr = np.argsort(-len_arr)
    else:
        idx_arr = np.arange(data_size)

    tail = idx_arr[len(idx_arr) - (len(idx_arr) % batch_size) :]
    idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % batch_size)]
    idx_arr = idx_arr.reshape((-1, batch_size)).tolist()
    if not drop_last and len(tail) > 0:
        idx_arr += [tail.tolist()]

    output = list()
    for idx in idx_arr:
   
        output.append(reprocess(data, idx))
   
    return output