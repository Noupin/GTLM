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
import utilities

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
        self.train_dataset = self.train_dataset.filter(utilities.filter_max_length)
        # cache the dataset to memory to get a speedup while reading from it.
        self.train_dataset = self.train_dataset.cache()
        self.train_dataset = self.train_dataset.shuffle(Tunable.tunableVars["BUFFER_SIZE"]).padded_batch(Tunable.tunableVars["BATCH_SIZE"])
        self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


        self.val_dataset = self.val_examples.map(self.tf_encode)
        self.val_dataset = self.val_dataset.filter(utilities.filter_max_length).padded_batch(Tunable.tunableVars["BATCH_SIZE"]) #Bringing the dataset to MAX_LENGTH

        self.pt_batch, self.en_batch = next(iter(self.val_dataset))


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

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


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