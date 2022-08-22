import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder
#from vocoder.hifigan import inference as vocoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)


    #print("\tTesting the encoder...")
    #encoder.embed_utterance(np.zeros(encoder.sampling_rate))
    #embed = np.random.rand(speaker_embedding_size)
    #embed /= np.linalg.norm(embed)
    #embeds = [embed, np.zeros(speaker_embedding_size)]
    #texts = ["test 1", "test 2"]
    #print("\tTesting the synthesizer... (loading the model will output a lot of text)")
    #mels = synthesizer.synthesize_spectrograms(texts, embeds)
    #mel = np.concatenate(mels, axis=1)
    #no_action = lambda *args: None
    #print("\tTesting the vocoder...")
    #vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
    #print("All test passed! You can now synthesize speech.\n\n")


    ## Interactive speech generation
    print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
          "show how you can interface this project easily with your own. See the source code for "
          "an explanation of what is happening.\n")

    print("Interactive generation loop")
    num_generated = 0

    # Get the reference audio filepath
    message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
                      "wav, m4a, flac, ...):\n"
    in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))

    ## Computing the embedding
    # First, we load the wav using the function that the speaker encoder provides. This is
    # important: there is preprocessing that must be applied.

    # The following two methods are equivalent:
    # - Directly load from the filepath:
    #preprocessed_wav = encoder.preprocess_wav(in_fpath)
    # - If the wav is already loaded:
    original_wav, sampling_rate = librosa.load(str(in_fpath))
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    print("Loaded file succesfully")
    # Then we derive the embedding. There are many functions and parameters that the
    # speaker encoder interfaces. These are mostly for in-depth research. You will typically
    # only use this function (with its default parameters):
    embed = encoder.embed_utterance(preprocessed_wav)
    print("Created the embedding")

    while True:
        try:
            ## Generating the spectrogram
            text = input("Write a sentence (+-20 words) to be synthesized:\n")

            # If seed is specified, reset torch seed and force synthesizer reload
            if args.seed is not None:
                torch.manual_seed(args.seed)
                synthesizer = Synthesizer(args.syn_model_fpath)

            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [text]
            embeds = [embed]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")


            ## Generating the waveform
            print("Synthesizing the waveform:")

            # If seed is specified, reset torch seed and reload vocoder
            if args.seed is not None:
                torch.manual_seed(args.seed)
                vocoder.load_model(args.voc_model_fpath)

            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            generated_wav = vocoder.infer_waveform(spec)


            ## Post-generation
            # There's a bug with sounddevice that makes the audio cut one second earlier, so we
            # pad it.
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
            
            # Trim excess silences to compensate for gaps in spectrograms (issue #53)
            #generated_wav = encoder.preprocess_wav(generated_wav)

            # Play the audio (non-blocking)
            if not args.no_sound:
                import sounddevice as sd
                try:
                    sd.stop()
                    sd.play(generated_wav, synthesizer.sample_rate)
                except sd.PortAudioError as e:
                    print("\nCaught exception: %s" % repr(e))
                    print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
                except:
                    raise

            # Save it on the disk
            filename = "demo_output_%02d.wav" % num_generated
            print(generated_wav.dtype)
            sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
            num_generated += 1
            print("\nSaved output as %s\n\n" % filename)


        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
