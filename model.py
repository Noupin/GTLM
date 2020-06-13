__author__ = "Noupin, TensorFlow"

#Third Party Imports
import os
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#First Party Import
from transformer import Transformer
from customSchedule import CustomSchedule
from tunable import Tunable
from preprocessing import Preprocessing

class Model():

    def __init__(self):
        #Optimizer
        '''
        Use the Adam optimizer with a custom learning rate scheduler
        '''

        self.prepro = Preprocessing()
        self.train_dataset = self.prepro.train_dataset

        self.learning_rate = CustomSchedule(Tunable.tunableVars["d_model"])

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, 
                                            epsilon=1e-9)

        #Show the learning rate graphically
        self.temp_learning_rate_schedule = CustomSchedule(Tunable.tunableVars["d_model"])

        #Loss & Metrics
        '''
        Since the target sequences are padded, it is important to apply a padding mask
        when calculating the loss.
        '''


        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        #Training & Checkpointing
        self.transformer = Transformer(Tunable.tunableVars["num_layers"], Tunable.tunableVars["d_model"], Tunable.tunableVars["num_heads"], Tunable.tunableVars["dff"],
                                self.prepro.input_vocab_size, self.prepro.target_vocab_size, 
                                pe_input=self.prepro.input_vocab_size, 
                                pe_target=self.prepro.target_vocab_size,
                                rate=Tunable.tunableVars["dropout_rate"])


        '''
        Create the checkpoint path and the checkpoint manager.
        This will be used to save checkpoints every n epochs.
        '''
        self.checkpoint_path = r"C:\Coding\Python\ML\Text\transformerCheckpoints"

        self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

        '''
        The target is divided into tar_inp and tar_real. tar_inp is passed as an input to the decoder.
        tar_real is that same input shifted by 1: At each location in tar_input, tar_real contains the
        next token that should be predicted.


        For example, sentence = "SOS A lion in the jungle is sleeping EOS"

        tar_inp = "SOS A lion in the jungle is sleeping"

        tar_real = "A lion in the jungle is sleeping EOS"


        The transformer is an auto-regressive model: it makes predictions one part at a time, and uses its
        output so far to decide what to do next.

        During training this example uses teacher-forcing. Teacher forcing is passing the true output to
        the next time step regardless of what the model predicts at the current time step.

        As the transformer predicts each word, self-attention allows it to look at the previous words in
        the input sequence to better predict the next word.

        To prevent the model from peeking at the expected output the model uses a look-ahead mask.
        '''

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64), tf.TensorSpec(shape=(None, None), dtype=tf.int64),])
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = self.prepro.create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, tar_inp, 
                                            True, 
                                            enc_padding_mask, 
                                            combined_mask, 
                                            dec_padding_mask)
            loss = self.prepro.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    #Evaluate
    '''
    The following steps are used for evaluation:

        - Encode the input sentence using the Portuguese tokenizer (tokenizer_pt). Moreover, add the
        start and end token so the input is equivalent to what the model is trained with. This is the encoder input.
        - The decoder input is the start token == tokenizer_en.vocab_size.
        - Calculate the padding masks and the look ahead masks.
        - The decoder then outputs the predictions by looking at the encoder output and its own
        output (self-attention).
        - Select the last word and calculate the argmax of that.
        - Concatentate the predicted word to the decoder input as pass it to the decoder.
        - In this approach, the decoder predicts the next word based on the previous words it predicted.

    Note: The model used here has less capacity to keep the example relatively faster so the
    predictions maybe less right. To reproduce the results in the paper, use the entire dataset
    and base transformer model or transformer XL, by changing the hyperparameters above.
    '''
    def evaluate(self, inp_sentence):
        start_token = [self.prepro.tokenizer_pt.vocab_size]
        end_token = [self.prepro.tokenizer_pt.vocab_size + 1]

        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + self.prepro.tokenizer_pt.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self.prepro.tokenizer_en.vocab_size]
        output = tf.expand_dims(decoder_input, 0)
        
        for _ in range(Tunable.tunableVars["MAX_LENGTH"]):
            enc_padding_mask, combined_mask, dec_padding_mask = self.prepro.create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input, 
                                                            output,
                                                            False,
                                                            enc_padding_mask,
                                                            combined_mask,
                                                            dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == self.prepro.tokenizer_en.vocab_size+1:
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(self, sentence, plot=''):
        result, attention_weights = self.evaluate(sentence)

        predicted_sentence = self.prepro.tokenizer_en.decode([i for i in result 
                                                if i < self.prepro.tokenizer_en.vocab_size])  

        print('Input: {}'.format(sentence))
        print('Predicted translation: {}'.format(predicted_sentence))

        if plot:
            self.prepro.plot_attention_weights(attention_weights, sentence, result, plot)
