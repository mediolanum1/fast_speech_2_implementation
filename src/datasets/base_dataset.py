import logging
import random

from textgrid import TextGrid
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset



logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    """
    Base class for the dataset
    """

    def __init__(self,
                index,
                dataset_part,
                cleaners,
                processed_dataset,
                target_sr=16000,
                text_encoder = None,
                limit=None,
                max_audio_length=None,
                max_text_length=None,
                shuffle_index=False,
                instance_transform=None,
                ):
        self._assert_index_is_valid(index)
        self._index: list[dict] = index
        self._dataset_part = dataset_part
        self._processed_dataset: dict = processed_dataset
        self.text_encoder = text_encoder
        self._cleaners = cleaners
        self.target_sr = target_sr
        self.energy_path = processed_dataset[f"{self._dataset_part}_energy_path"]
        self.pitch_path = processed_dataset[f"{self._dataset_part}_pitch_path"]
        self.mel_spectrogram_path = processed_dataset[f"{self._dataset_part}_mel_path"]
        self.transcripts_path = processed_dataset[f"transcripts"]
    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.
    
        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.
    
        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        basename = data_dict["basename"]
        audio_path = data_dict["path"]
        
        audio, audio_len = self.load_audio(audio_path)
     
        text = data_dict["text"]
        text_encoded =  np.array(text_to_sequence(text, self._cleaners))
        
        mel_path = os.path.join(
            self.mel_spectrogram_path,
            "{}_mel.npy".format(basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.pitch_path,
            "{}_pitch.npy".format(basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.energy_path,
            "{}_energy.npy".format(basename),
        )
        energy = np.load(energy_path)

        phonemes_path = os.path.join(
            self.transcripts_path,
            "{}.TextGrid".format(basename),
        )
        phonemes, durations= self.extract_phonemes_and_durations(phonemes_path)
    
        instance_data = {
            "audio": audio,
            "text": text,
            #"phonemes_encoded": phonemes_encoded,
            "phonemes": phonemes,
            "durations": durations,
            "text_encoded": text_encoded,
            "audio_path": audio_path,
            "audio_len": audio_len,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
        }
        return instance_data
    
    def __len__(self):
        """ 
        Get length of the dataset (lenght of the index).
        """
        return len(self._index)
        
    def extract_phonemes_and_durations(self,textgrid_path, tier_name='phones'):
        tg = TextGrid.fromFile(textgrid_path)
        
        for tier in tg.tiers:
            if tier.name == tier_name:
                phonemes = []
                durations = []
             #   phonemes_encoded = []
                # Extract intervals with non-empty phonemes
                for interval in tier:
                    phoneme = interval.mark.strip()
                    if phoneme:
                        start = interval.minTime
                        end = interval.maxTime
                        duration = end - start
                      #  phonemes_encoded.append(text_to_sequence(phoneme, self._cleaners))
                        phonemes.append(phoneme)
                        durations.append(duration)
                        
                return phonemes, np.array(durations)
        

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_length_seconds = audio_tensor.size(1) / self.target_sr
        audio_tensor = audio_tensor[0:1, :]  
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)

        return audio_tensor,audio_length_seconds
            
    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.
    
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
            assert "text" in entry, (
                "Each dataset item should include field 'text'"
                " - object ground-truth transcription."
            )
    
