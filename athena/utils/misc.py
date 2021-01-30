# coding=utf-8
# Copyright (C) ATHENA AUTHORS
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
# pylint: disable=missing-function-docstring, invalid-name
""" misc """
import os
import wave
import tensorflow as tf
from absl import logging
import numpy as np


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def create_input_masks(x, x_length):
    """Generate a square mask for the sequence for mult-head attention.
       The masked positions are filled with float(1.0).
       Unmasked positions are filled with float(0.0).
    """
    return tf.sequence_mask(
        x_length, tf.shape(x)[1], dtype=tf.float32)

def create_input_paddings(x, x_length):
    """Generate a square mask for the sequence for mult-head attention.
       The masked positions are filled with float(1.0).
       Unmasked positions are filled with float(0.0).
    """
    return 1. - create_input_masks(x, x_length)

def mask_index_from_labels(labels, index):
    mask = tf.math.logical_not(tf.math.equal(labels, index))
    mask = tf.cast(mask, dtype=labels.dtype)
    return labels * mask


def insert_sos_in_labels(labels, sos):
    sos = tf.ones([tf.shape(labels)[0], 1], dtype=labels.dtype) * sos
    return tf.concat([sos, labels], axis=-1)


def remove_eos_in_labels(input_labels, labels_length):
    """remove eos in labels, batch size should be larger than 1
    assuming 0 as the padding and the last one is the eos
    """
    labels = input_labels[:, :-1]
    length = labels_length - 1
    max_length = tf.shape(labels)[1]
    mask = tf.sequence_mask(length, max_length, dtype=labels.dtype)
    labels = labels * mask
    labels.set_shape([None, None])
    return labels


def insert_eos_in_labels(input_labels, eos, labels_length):
    """insert eos in labels, batch size should be larger than 1
    assuming 0 as the padding,
    """
    zero = tf.zeros([tf.shape(input_labels)[0], 1], dtype=input_labels.dtype)
    labels = tf.concat([input_labels, zero], axis=-1)
    labels += tf.one_hot(labels_length, tf.shape(labels)[1], dtype=labels.dtype) * eos
    return labels


def generate_square_subsequent_mask(size):
    """Generate a square mask for the sequence. The masked positions are filled with float(1.0).
       Unmasked positions are filled with float(0.0).
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_multihead_mask(x, x_length, y, reverse=False):
    """Generate a square mask for the sequence for mult-head attention.
       The masked positions are filled with float(1.0).
       Unmasked positions are filled with float(0.0).
    """
    x_mask, y_mask = None, None
    if x is not None:
        x_mask = 1.0 - tf.sequence_mask(
            x_length, tf.shape(x)[1], dtype=tf.float32
        )
        x_mask = x_mask[:, tf.newaxis, tf.newaxis, :]
        if reverse:
            look_ahead_mask = generate_square_subsequent_mask(tf.shape(x)[1])
            x_mask = tf.maximum(x_mask, look_ahead_mask)
        x_mask.set_shape([None, None, None, None])
    if y is not None:
        y_mask = tf.cast(tf.math.equal(y, 0), tf.float32)
        y_mask = y_mask[:, tf.newaxis, tf.newaxis, :]
        if not reverse:
            look_ahead_mask = generate_square_subsequent_mask(tf.shape(y)[1])
            y_mask = tf.maximum(y_mask, look_ahead_mask)
        y_mask.set_shape([None, None, None, None])
    return x_mask, y_mask


def create_multihead_y_mask(y, reverse=False):
    """Generate a square mask for the sequence for mult-head attention.
       The masked positions are filled with float(1.0).
       Unmasked positions are filled with float(0.0).
    """
    y_mask = tf.cast(tf.math.equal(y, 0), tf.float32)
    y_mask = y_mask[:, tf.newaxis, tf.newaxis, :]
    if not reverse:
        look_ahead_mask = generate_square_subsequent_mask(tf.shape(y)[1])
        y_mask = tf.maximum(y_mask, look_ahead_mask)
    y_mask.set_shape([None, None, None, None])
    return y_mask


def gated_linear_layer(inputs, gates, name=None):
    h1_glu = tf.keras.layers.multiply(inputs=[inputs, tf.sigmoid(gates)], name=name)
    return h1_glu


def validate_seqs(seqs, eos):
    """Discard end symbol and elements after end symbol

    Args:
        seqs: shape=(batch_size, seq_length)
        eos: eos id

    Returns:
        validated_preds: seqs without eos id
    """
    eos = tf.cast(eos, tf.int64)
    if tf.shape(seqs)[1] == 0:
        validated_preds = tf.zeros([tf.shape(seqs)[0], 1], dtype=tf.int64)
    else:
        if eos != 0:
            indexes = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
            a = tf.not_equal(eos, seqs)
            res = a[:, 0]
            indexes = indexes.write(0, res)
            for i in tf.range(1, tf.shape(a)[1]):
                res = tf.logical_and(res, a[:, i])
                indexes = indexes.write(i, res)
            res = tf.transpose(indexes.stack(), [1, 0])
            validated_preds = tf.where(tf.logical_not(res), tf.zeros_like(seqs), seqs)
        else:
            validated_preds = tf.where(tf.equal(eos, seqs), tf.zeros_like(seqs), seqs)
    validated_preds = tf.sparse.from_dense(validated_preds)
    counter = tf.cast(tf.shape(validated_preds.values)[0], tf.float32)
    return validated_preds, counter


def get_wave_file_length(wave_file):
    """get the wave file length(duration) in ms

    Args:
        wave_file: the path of wave file

    Returns:
        wav_length: the length(ms) of the wave file
    """
    if not os.path.exists(wave_file):
        logging.warning("Wave file {} does not exist!".format(wave_file))
        return 0
    with wave.open(wave_file) as wav_file:
        wav_frames = wav_file.getnframes()
        wav_frame_rate = wav_file.getframerate()
        wav_length = int(wav_frames / wav_frame_rate * 1000)  # get wave duration in ms
    return wav_length


def splice_numpy(x, context):
    """
    Splice a tensor along the last dimension with context.
    
    Example:

    >>> t = [[[1, 2, 3],
    >>>     [4, 5, 6],
    >>>     [7, 8, 9]]]
    >>> splice_tensor(t, [0, 1]) =
    >>>   [[[1, 2, 3, 4, 5, 6],
    >>>     [4, 5, 6, 7, 8, 9],
    >>>     [7, 8, 9, 7, 8, 9]]]

    Args:
        tensor: a tf.Tensor with shape (B, T, D) a.k.a. (N, H, W)
        context: a list of context offsets

    Returns:
        spliced tensor with shape (..., D * len(context))
    """
    # numpy can speed up 10%
    x = x.numpy()
    input_shape = np.shape(x)
    B, T = input_shape[0], input_shape[1]
    context_len = len(context)
    left_boundary = -1 * min(context) if min(context) < 0 else 0
    right_boundary = max(context) if max(context) > 0 else 0
    sample_range = ([0] * left_boundary + [i for i in range(T)] + [T - 1] * right_boundary)
    array = []
    for idx in range(context_len):
        pos = context[idx]
        if pos < 0:
            pos = len(sample_range) - T - max(context) + pos
        sliced = x[:, sample_range[pos : pos + T], :]
        array += [sliced]
    spliced = np.concatenate([i[:, :, np.newaxis, :] for i in array], axis=2)
    spliced = np.reshape(spliced, (B, T, -1))
    return tf.convert_to_tensor(spliced)

def set_default_summary_writer(summary_directory=None):
    if summary_directory is None:
        summary_directory = os.path.join(os.path.expanduser("~"), ".athena")
        summary_directory = os.path.join(summary_directory, "event")
    writer = tf.summary.create_file_writer(summary_directory)
    writer.set_as_default()

def tensor_shape(tensor):
    """Return a list with tensor shape. For each dimension,
       use tensor.get_shape() first. If not available, use tf.shape().
    """
    if tensor.get_shape().dims is None:
        return tf.shape(tensor)
    shape_value = tensor.get_shape().as_list()
    shape_tensor = tf.shape(tensor)
    ret = [shape_tensor[idx]
        if shape_value[idx] is None
        else shape_value[idx]
        for idx in range(len(shape_value))]
    return ret


def apply_label_smoothing(inputs, num_classes, smoothing_rate=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      num_classes: Number of classes.
      smoothing_rate: Smoothing rate.
    ```
    '''
    return ((1.0 - smoothing_rate) * inputs) + (smoothing_rate / num_classes)

def subsequent_chunk_mask(size, chunk_size, history_chunk_size=-1):
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        history_chunk_size (int): size of history chunk

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(4, 2, 1)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [0, 1, 1, 1],
         [0, 1, 1, 1]]
    """
    '''
    ret = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, 0:ending] = 1
        if history_chunk_size != -1:
            padding_zero_ending = max(0, (i // chunk_size)* chunk_size-history_chunk_size)
            ret[i, 0:padding_zero_ending] = 0.0
    return ret
    '''
    x = tf.broadcast_to(
        tf.cast(tf.range(size), dtype=tf.int32), (size, size))

    chunk_current = tf.math.floordiv(x, chunk_size) * chunk_size
    chunk_ending = tf.math.minimum(chunk_current+chunk_size, size)
    chunk_ending_ = tf.transpose(chunk_ending)
    ret = tf.cast(tf.math.less_equal(
        chunk_ending, chunk_ending_), dtype=tf.float32)
    if history_chunk_size != -1:
        chunk_start = tf.math.maximum(
            0, tf.transpose(chunk_current)-history_chunk_size)
        history_mask = tf.cast(
            tf.math.greater_equal(x, chunk_start), dtype=tf.float32)
        ret = history_mask*ret

    return ret


def add_optional_chunk_mask(xs, masks, training,
                            use_dynamic_chunk: bool, decoding_chunk_size: int,
                            static_chunk_size: int, history_chunk_size: int = -1,
                            dynamic_max_chunk: int = 25, dynamic_include_fullcontext: bool = True):
    """ Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        decoding_chunk_size (int): decoding chunk size for dynamic chunk.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    # masks = ~make_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, L)
    # (B, L) -> (B, 1, L)
    masks = tf.expand_dims(masks, 1)
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = shape_list(xs)[1]
        chunk_size = tf.random.uniform(
            shape=(), minval=1, maxval=max_len, dtype=tf.int32)

        if dynamic_include_fullcontext:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.

            '''
            chunk_size = torch.randint(1, max_len, (1, )).item()
            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
            '''
            chunk_size = tf.cond(
                tf.math.greater(chunk_size, tf.math.floordiv(max_len, 2)),
                lambda: max_len,
                lambda: tf.math.floormod(chunk_size, dynamic_max_chunk) + 1
            )
        else:
            # chunk size is either [1, 25].
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = tf.math.floormod(
                chunk_size, dynamic_max_chunk) + 1

        # Use chunk 1 when testing in dynamic chunk mode.
        chunk_size = tf.cond(training,
                             lambda: chunk_size,
                             lambda: decoding_chunk_size)

        # history chunk size is either [1, 25] or full context.
        # Since we use 4 times subsampling and allow up to 1s(100 frames)
        # delay, the maximum frame is 100 / 4 = 25.
        chunk_masks = subsequent_chunk_mask(
            shape_list(xs)[1], chunk_size, history_chunk_size)  # (L, L)

        # chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = tf.expand_dims(chunk_masks, 0)
        chunk_masks = masks * chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        # history chunk size is either [1, 25] or full context.
        # Since we use 4 times subsampling and allow up to 1s(100 frames)
        # delay, the maximum frame is 100 / 4 = 25.
        chunk_masks = subsequent_chunk_mask(
            shape_list(xs)[1], static_chunk_size, history_chunk_size)  # (L, L)

        # chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = tf.expand_dims(chunk_masks, 0)
        chunk_masks = masks * chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks

def create_multihead_chunk_mask(xs, seq_len, training,
                                use_dynamic_chunk: bool, decoding_chunk_size: int,
                                static_chunk_size: int = 0, history_chunk_size: int = -1,
                                dynamic_max_chunk: int = 25, dynamic_include_fullcontext: bool = True):

    # (B, L)
    masks = 1.0 - create_input_paddings(xs, seq_len)

    # (B, L, L)
    chunk_masks = add_optional_chunk_mask(xs, masks, training,
                                          use_dynamic_chunk, decoding_chunk_size,
                                          static_chunk_size, history_chunk_size=history_chunk_size,
                                          dynamic_max_chunk=dynamic_max_chunk,
                                          dynamic_include_fullcontext=dynamic_include_fullcontext)
    chunk_masks = 1.0-chunk_masks

    # (B, 1, L, L)
    return tf.expand_dims(chunk_masks, 1)

