__author__ = "Noupin, TensorFlow"

#Third Party Imports
import os
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#First Party Import
from customSchedule import CustomSchedule
from tunable import Tunable
from constants import Constants

class Preprocessing():
    def __init__(self):
        #Loading the data
        self.examples, self.metadata = tfds.load('ted_hrlr_translate/{langShort}_to_en'.format(langShort=Constants.languageMap[Tunable.tunableVars["language"]]), 
                                                 with_info=True,
                                                 as_supervised=True)
        self.train_examples, self.val_examples = self.examples['train'], self.examples['validation']

        #Creating subword Tokenizer from the training datset
        #The tokenizer encodes the string by breaking it into subwords if the word is not in its dictionary

        #English
        if not os.path.exists(Constants.tokenizerPath + "english.subwords"):
            print("Creating English Subword Tokenizer.")
            self.tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                        (en.numpy() for pt, en in self.train_examples), target_vocab_size=2**13)
            self.tokenizer_en.save_to_file(Constants.tokenizerPath + "english")
            print("Finsihed Creating English Subword Tokenizer")
        else:
            print("Loading English Subword Tokenizer.")
            self.tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(Constants.tokenizerPath + "english")
            print("Finsihed Loading English Subword Tokenizer")


        #Portuguese
        if not os.path.exists(Constants.tokenizerPath + "{lang}.subwords".format(lang=Tunable.tunableVars["language"])):
            print("Creating {lang} Subword Tokenizer.".format(lang=Tunable.tunableVars["language"].title()))
            self.tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                        (pt.numpy() for pt, en in self.train_examples), target_vocab_size=2**13)
            self.tokenizer_pt.save_to_file(Constants.tokenizerPath + Tunable.tunableVars["language"])
            print("Finsihed Creating {lang} Subword Tokenizer".format(lang=Tunable.tunableVars["language"].title()))
        else:
            print("Loading {lang} Subword Tokenizer.".format(lang=Tunable.tunableVars["language"].title()))
            self.tokenizer_pt = tfds.features.text.SubwordTextEncoder.load_from_file(Constants.tokenizerPath + Tunable.tunableVars["language"])
            print("Finsihed Loading {lang} Subword Tokenizer".format(lang=Tunable.tunableVars["language"].title()))



        self.train_dataset = self.train_examples.map(self.tf_encode)
        self.train_dataset = self.train_dataset.filter(self.filter_max_length)
        # cache the dataset to memory to get a speedup while reading from it.
        self.train_dataset = self.train_dataset.cache()
        self.train_dataset = self.train_dataset.shuffle(Tunable.tunableVars["BUFFER_SIZE"]).padded_batch(Tunable.tunableVars["BATCH_SIZE"])
        self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


        self.val_dataset = self.val_examples.map(self.tf_encode)
        self.val_dataset = self.val_dataset.filter(self.filter_max_length).padded_batch(Tunable.tunableVars["BATCH_SIZE"]) #Bringing the dataset to MAX_LENGTH

        self.pt_batch, self.en_batch = next(iter(self.val_dataset))


        #Hyperparameters
        '''
        To keep this example small and relatively fast, the values for num_layers, d_model, and dff have been reduced.

        The values used in the base model of transformer were; num_layers=6, d_model = 512, dff = 2048.

        Note: By changing the values below, you can get the model that achieved state of the art on many tasks.
        '''

        self.input_vocab_size = self.tokenizer_pt.vocab_size + 2
        self.target_vocab_size = self.tokenizer_en.vocab_size + 2

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    
    #Adding a start and end token to the input target
    def encode(self, lang1, lang2):
        lang1 = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            lang1.numpy()) + [self.tokenizer_pt.vocab_size+1]
        
        lang2 = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            lang2.numpy()) + [self.tokenizer_en.vocab_size+1]
        
        return lang1, lang2

    '''
    You want to use Dataset.map to apply this function to each element of the dataset.
    Dataset.map runs in graph mode.

    Graph tensors do not have a value.
    In graph mode you can only use TensorFlow Ops and functions.
    So you can't .map this function directly: You need to wrap it in a tf.py_function.
    The tf.py_function will pass regular tensors (with a value and a .numpy() method to access it), to the wrapped python function.
    '''
    def tf_encode(self, pt, en):
        result_pt, result_en = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en

    #To keep this example small and relatively fast, drop examples with a length of over 40 tokens
    def filter_max_length(self, x, y, max_length=Tunable.tunableVars["MAX_LENGTH"]):
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
    def get_angles(self,  pos, i, d_model):
        angle_rates = 1/np.power(10000, (2* (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
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
    def create_padding_mask(self, seq):
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
    def create_look_ahead_mask(self, size):
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
    def scaled_dot_product_attention(self, q, k, v, mask):
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

    def print_out(self, q, k, v):
        temp_out, temp_attn = self.scaled_dot_product_attention(
            q, k, v, None)
        print ('Attention weights are:')
        print (temp_attn)
        print ('Output is:')
        print (temp_out)

    '''
    Point wise feed forward network consists of two fully-connected
    layers with a ReLU activation in between.
    '''
    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'), #(batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model) #(batch_size, seq_len, d_model)
        ])

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by 
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def plot_attention_weights(self, attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))

        sentence = self.tokenizer_pt.encode(sentence)

        attention = tf.squeeze(attention[layer], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head+1)

            # plot the attention weights
            ax.matshow(attention[head][:-1, :], cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(sentence)+2))
            ax.set_yticks(range(len(result)))

            ax.set_ylim(len(result)-1.5, -0.5)
                
            ax.set_xticklabels(
                ['<start>']+[self.tokenizer_pt.decode([i]) for i in sentence]+['<end>'], 
                fontdict=fontdict, rotation=90)

            ax.set_yticklabels([self.tokenizer_en.decode([i]) for i in result 
                                if i < self.tokenizer_en.vocab_size], 
                                fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head+1))

        plt.tight_layout()
        plt.show()