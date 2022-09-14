import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
# from vocoder import inference as vocoder
from vocoder.hifigan import inference as vocoder
from synthesizer.inference import Synthesizer
from utils.dsp import DSP

from configure import config
from app import utils


# from vocoder.hifigan import inference as vocoder

class VoiceClone(object):

    def __init__(self):
        print("Preparing the encoder, the synthesizer and the vocoder...")
        self.seed = None
        encoder.load_model(Path("saved_models/default/encoder.pt"))
        self.synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
        vocoder.load_model(Path("saved_models/hifi/vocoder.pt"))
        print("Init Complete")

    def create_wave(self, message_id, demo_wav, query):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        print("Running a test of your configuration...\n")

        original_wav, sampling_rate = librosa.load(str(demo_wav))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        print("Loaded file succesfully")
        embed = encoder.embed_utterance(preprocessed_wav)
        print("Created the embedding")

        try:
            if self.seed is not None:
                torch.manual_seed(args.seed)
                synthesizer = Synthesizer(args.syn_model_fpath)

            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [query]
            embeds = [embed]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")

            ## Generating the waveform
            print("Synthesizing the waveform:")

            # If seed is specified, reset torch seed and reload vocoder
            if self.seed is not None:
                torch.manual_seed(args.seed)
                vocoder.load_model(args.voc_model_fpath)

            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            generated_wav = vocoder.infer_waveform(spec)

            ## Post-generation
            # There's a bug with sounddevice that makes the audio cut one second earlier, so we
            # pad it.
            generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")

            # Save it on the disk
            filename = os.path.join(config.OUT_AUDIO, f"demo_output_{message_id}_{utils.get_timestamp()}.wav")
            sf.write(filename, generated_wav.astype(np.float32), self.synthesizer.sample_rate)
            print("\nSaved output as %s\n\n" % filename)
            return filename

        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
