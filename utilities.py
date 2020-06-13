__author__ = "Noupin, TensorFlow"

#Third Party Imports
import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import json
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

#First Party Import
from tunable import Tunable


#To keep this example small and relatively fast, drop examples with a length of over 40 tokens
def filter_max_length(x, y, max_length=Tunable.tunableVars["MAX_LENGTH"]):
    return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

#Positional Encoding
'''
Since this model doesn't contain any recurrence or convolution, positional encoding
is added to give the model some information about the relative position of the words in the sentence.
The positional encoding vector is added to the embedding vector.
Embeddings represent a token in a d-dimensional space where tokens with
similar meaning will be closer to each other. But the embeddings do not
encode the relative position of words in a sentence. So after adding the
positional encoding, words will be closer to each other based on the similarity
of their meaning and their position in the sentence, in the d-dimensional space.
'''
def get_angles(pos, i, d_model):
    angle_rates = 1/np.power(10000, (2* (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

#Masking
'''
Mask all the pad tokens in the batch of sequence. It ensures that the
model does not treat padding as the input. The mask indicates where pad
value 0 is present: it outputs a 1 at those locations, and a 0 otherwise.
'''
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    #add extra dimensions to add the padding to the attention logits
    return seq[:, tf.newaxis, tf.newaxis, :] #(batch_size, 1, 1, seq_len)

'''
The look-ahead mask is used to mask the future tokens in a sequence.
In other words, the mask indicates which entries should not be used.
This means that to predict the third word, only the first and second
word will be used. Similarly to predict the fourth word, only the first,
second and the third word will be used and so on. User for the masked
multi-head attention block.
'''
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask #(seq_len, seq_len)

#Scaled Dot Product Attention
'''
The attention function used by the transformer takes three inputs: Q (query), K (key), V (value).
The dot-product attention is scaled by a factor of square root of the depth. This is done because
for large values of depth, the dot product grows large in magnitude pushing the softmax function
where it has small gradients resulting in a very hard softmax.
For example, consider that Q and K have a mean of 0 and variance of 1. Their matrix multiplication
will have a mean of 0 and variance of dk. Hence, square root of dk is used for scaling (and not any other number)
because the matmul of Q and K should have a mean of 0 and variance of 1, and you get a gentler softmax.
The mask is multiplied with -1e9 (close to negative infinity). This is done because the mask is summed
with the scaled matrix multiplication of Q and K and is applied immediately before a softmax. The goal
is to zero out these cells, and large negative inputs to softmax are near zero in the output.
'''
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True) #(..., seq_len_q, seq_len_k)

    #scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    #add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) #(..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v) #(..., seq_len_q, depth_v)

    return output, attention_weights

'''
As the softmax normalization is done on K, its values decide the amount of importance given to Q.
The output represents the multiplication of the attention weights and the V (value) vector.
This ensures that the words you want to focus on are kept as-is and the irrelevant words are flushed out.
'''

def print_out( q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None)
    print ('Attention weights are:')
    print (temp_attn)
    print ('Output is:')
    print (temp_out)

'''
Point wise feed forward network consists of two fully-connected
layers with a ReLU activation in between.
'''
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'), #(batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model) #(batch_size, seq_len, d_model)
    ])

def create_masks(inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by 
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask