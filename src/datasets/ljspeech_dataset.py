import json
import os
import shutil
from pathlib import Path
import numpy as np
import torchaudio
import wget
from tqdm import tqdm
import torch
import librosa

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


URL_LINKS = {
  "ljspeech":  "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
}


class LJSpeechDataset(BaseDataset):
    def __init__(self, cleaners, n_fft, n_mels, win_length, hop_length, fmin, fmax, sr, data_dir=None, *args, **kwargs):
        self.name = "LJSpeechDataset"
        if data_dir is None: 
            data_dir = ROOT_PATH / "working" / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self.cleaners = cleaners
        self.sr = sr  
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.part = "ljspeech"
        self.index = self._get_or_load_index("ljspeech")
        processed_dataset = {f"{self.part}_mel_path": self.mel_path, f"{self.part}_energy_path": self.energy_path, f"{self.part}_pitch_path": self.pitch_path, f"transcripts": self.transcripts_path }
        super().__init__(self.index,self.part,self.cleaners, processed_dataset, *args, **kwargs)
    
    def __len__(self):
        """
        Get length of the dataset (lenght of the index).
        """
        return len(self.index)
        
    def _get_or_load_index(self, part):
        self.mel_path = self._data_dir / f"processed_dataset" / f"{part}_mel"
        self.pitch_path = self._data_dir / f"processed_dataset" / f"{part}_pitch"
        self.energy_path = self._data_dir / f"processed_dataset" / f"{part}_energy"
        self.transcripts_path = self._data_dir / "transcripts"

        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists() and self.mel_path.exists() and self.energy_path.exists() and self.pitch_path.exists() and self.transcripts_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            if not index_path.exists():
                index = self._create_index(part)
                with index_path.open("w") as f:
                    json.dump(index, f, indent=2)
            if not self.mel_path.exists():
                print("mel spectrogramms are not present, creating")
                self._create_mel_specs(part)
        
        return index
        
    def _create_mel_specs(self,part):
        mel_basis = librosa_mel_fn(
            sr= self.sr,n_fft= self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax
        )
        self.mel_basis = torch.from_numpy(mel_basis).float()

        total_rows = sum(1 for _ in open(self._data_dir / "metadata.csv", "r", encoding="utf-8")) - 1
        with open(os.path.join(self._data_dir, "metadata.csv"), encoding="utf-8") as f:
            for line in tqdm(f, total=total_rows, desc="processing metadata.cvs for mel spectrograms"):
                parts = line.strip().split("|")
                base_name = parts[0]
                text = self._clean_text(parts[2])
                wav_path = os.path.join(self._data_dir, "wavs", "{}.wav".format(base_name))
                mel_spec_path = self._process_wav_to_mel(wav_path, base_name, part, output_folders = [f"{part}_mel",f"{part}_pitch",f"{part}_energy"])
                
    def _load_part(self, part):
        if (self._data_dir / "metadata.csv").exists() and (self._data_dir / "wavs").exists():
            print(f"Part {part} already exists. Skipping download.")
            return
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        wget.download(URL_LINKS[part], str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))
        
    def _clean_text(self, text):
        processed_text = text
        for cleaner in self.cleaners.values(): 
            processed_text = cleaner(processed_text)
        return processed_text
        
    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)
        transcript_data_dir = self._data_dir / "transcripts"
        if not transcript_data_dir.exists():  
            transcript_data_dir.mkdir(exist_ok=True, parents=True)
        texts_data_dir = self._data_dir / "texts"
        if not texts_data_dir.exists():  
            texts_data_dir.mkdir(exist_ok=True, parents=True)
        flac_dirs = set()
        total_rows = sum(1 for _ in open(self._data_dir / "metadata.csv", "r", encoding="utf-8")) - 1
        with open(os.path.join(self._data_dir, "metadata.csv"), encoding="utf-8") as f:
            for line in tqdm(f, total=total_rows, desc="processing metadata.cvs"):
                parts = line.strip().split("|")
                basename = parts[0]
                text = self._clean_text(parts[2])
                wav_path = os.path.join(self._data_dir, "wavs", "{}.wav".format(basename))
                self.create_textgrid(basename,wav_path, text, transcript_data_dir, texts_data_dir)
                if os.path.exists(wav_path):
                    index.append(   
                           {
                                "basename": basename,
                                "path": wav_path,
                                "text": text.lower(),
                          
                            }
                    )
        return index

    def create_textgrid(self, wav_filename,wav_path ,transcript, output_dir,text_output, speaker_id="ljspeech"):
        """
        Generate a Praat TextGrid file for a given wav file and transcript.
    
        Args:
            wav_filename (str): Name of the WAV file.
            transcript (str): Corresponding transcript text.
            output_dir (str): Directory to save the TextGrid file.
            speaker_id (str): Name of the tier (optional speaker ID).
            
        """
        wav, sr = torchaudio.load(wav_path)
        audio_len  = wav.size(1) / self.sr
        textgrid_content = f"""File type = "ooTextFile"
        Object class = "TextGrid"
        
        xmin = 0
        xmax = {audio_len}
        tiers? <exists>
        size = 1
        item []:
            item [1]:
                class = "IntervalTier"
                name = "{speaker_id}"
                xmin = 0
                xmax = {audio_len}
                intervals: size = 1
                intervals [1]:
                    xmin = 0
                    xmax = {audio_len}
                    text = "{transcript}"
        """

        textgrid_filename = wav_filename + ".TextGrid"
        textgrid_path = os.path.join(output_dir, textgrid_filename)
        with open(textgrid_path, "w") as f:
            f.write(textgrid_content)

        text_filename = wav_filename + ".txt"
        text_path = os.path.join(text_output, text_filename)
        with open(text_path, "w") as g: 
            g.write(transcript)
     
    def _process_wav_to_mel(self, wav_path, base_name, part, output_folders):
        for output_folder in output_folders:
            folder_path = self._data_dir/ f"processed_dataset" / output_folder
            if not os.path.exists(folder_path):
                folder_path.mkdir(exist_ok=True, parents=True)
        y, sr = librosa.load(wav_path, sr=self.sr, mono=True)
       
        # Pitch calculation (using Harmonic-to-Noise Ratio for simplicity)
        def calculate_pitch(signal, hop_length, sr):
            # Compute the pitch for the entire signal
            pitches, magnitudes = librosa.core.piptrack(y=signal, sr=sr, n_fft=self.n_fft, hop_length=hop_length)
            # Extract pitch from the max magnitude values (vectorized)
            pitch = pitches[magnitudes.argmax(axis=0), np.arange(magnitudes.shape[1])]
            return pitch       
            
        def spectral_normalize(x,C=1, clip_val=1e-5): 
            """
            PARAMS
            ------
            C: compression factor
            """
            return torch.log(torch.clamp(x, min=clip_val) * C)
        pitch = calculate_pitch(y,  hop_length=self.hop_length, sr=self.sr) 
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window="hann")
        magnitudes, phases = librosa.magphase(D)
        magnitudes = torch.tensor(magnitudes) 
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = spectral_normalize(mel_output)
        energy = torch.norm(magnitudes, dim=1)
            
        np.save(os.path.join(self._data_dir / f"processed_dataset",output_folders[0], f"{base_name}_mel"), mel_output)
        np.save(os.path.join(self._data_dir / f"processed_dataset",output_folders[1], f"{base_name}_pitch"), pitch)      
        np.save(os.path.join(self._data_dir / f"processed_dataset",output_folders[2], f"{base_name}_energy"), energy) 




