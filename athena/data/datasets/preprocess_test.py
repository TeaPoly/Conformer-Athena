# coding=utf-8
# Lekai Huang;
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=no-member, invalid-name

import librosa
import tensorflow as tf

# for visualizing first one result of SpecAugment
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from preprocess import SpecAugment


def visualization_spectrogram(mel_spectrogram, title):
    """visualizing first one result of SpecAugment
    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(
        mel_spectrogram[:, :], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load an audio file as a floating point time series.
    audio, sampling_rate = librosa.load("test.wav")

    # Compute a mel-scaled spectrogram.
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=sampling_rate,
                                                     n_mels=256,
                                                     hop_length=128,
                                                     fmax=8000)

    # visualization_spectrogram(mel_spectrogram, "mel_spectrogram")
    # (frequecy, time) -> (time, frequecy)
    mel_spectrogram = mel_spectrogram.transpose()


    # Inserts a dimension of 1 into a tensor's shape.
    # (time, frequecy) -> (time, frequecy, channel)
    mel_spectrogram = mel_spectrogram.reshape(
        (mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1))

    config = {
        "time_warping": 80,
        "time_masking": 40,
        "frequency_masking": 30,
        "mask_cols": 2,
        "mask_type": "zeros"
    }

    specaug = SpecAugment(config)

    # (time, frequecy, channel)
    warped_masked_spectrogram = specaug(
        tf.convert_to_tensor(mel_spectrogram)
    )

    # visualizing first one result of SpecAugment
    warped_masked_spectrogram = warped_masked_spectrogram.numpy()
    warped_masked_spectrogram = warped_masked_spectrogram.reshape(
        (warped_masked_spectrogram.shape[0], warped_masked_spectrogram.shape[1]))
    warped_masked_spectrogram = tf.transpose(warped_masked_spectrogram)
    visualization_spectrogram(warped_masked_spectrogram,
                              "warped_masked_spectrogram")
