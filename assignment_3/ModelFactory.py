import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, Embedding, Dense, Flatten, LayerNormalization, MultiHeadAttention, Dropout, Conv1D
from tensorflow.keras.preprocessing.text import Tokenizer



# Files with English to Devanagari (Hindi) translation word by word
# Punctutations have already been cleaned from this file

def get_data_files(language):
    """ Function fo read data
    """

    ## REPLACE THIS PATH UPTO dakshina_dataset_v1.0 with your own dataset path ##
    template = "./DakshinaDataset/dakshina_dataset_v1.0/{}/lexicons/{}.translit.sampled.{}.tsv"

    train_tsv = template.format(language, language, "train")
    val_tsv = template.format(language, language, "dev")
    test_tsv = template.format(language, language, "test")

    return train_tsv, val_tsv, test_tsv


## Utility functions for preprocessing data ##

def add_start_end_tokens(df, cols, sos="\t", eos="\n"):
    """ Adds EOS and SOS tokens to data
    """

    def add_tokens(s):
        # \t = starting token
        # \n = ending token
        return sos + str(s) + eos

    for col in cols:
        df[col] = df[col].apply(add_tokens)


def tokenize(lang, tokenizer=None):
    """ Uses tf.keras tokenizer to tokenize the data/words into characters
    """

    if tokenizer is None:
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(lang)

        lang_tensor = tokenizer.texts_to_sequences(lang)
        lang_tensor = tf.keras.preprocessing.sequence.pad_sequences(lang_tensor,
                                                                    padding='post')

    else:
        lang_tensor = tokenizer.texts_to_sequences(lang)
        lang_tensor = tf.keras.preprocessing.sequence.pad_sequences(lang_tensor,
                                                                    padding='post')

    return lang_tensor, tokenizer


def preprocess_data(fpath, input_lang_tokenizer=None, targ_lang_tokenizer=None):
    """ Reads, tokenizes and adds SOS/EOS tokens to data based on above functions
    """

    df = pd.read_csv(fpath, sep="\t", header=None)

    # adding start and end tokens to know when to stop predicting
    add_start_end_tokens(df, [0, 1])

    input_lang_tensor, input_tokenizer = tokenize(df[1].astype(str).tolist(),
                                                  tokenizer=input_lang_tokenizer)

    targ_lang_tensor, targ_tokenizer = tokenize(df[0].astype(str).tolist(),
                                                tokenizer=targ_lang_tokenizer)

    dataset = tf.data.Dataset.from_tensor_slices((input_lang_tensor, targ_lang_tensor))
    dataset = dataset.shuffle(len(dataset))

    return dataset, input_tokenizer, targ_tokenizer



def getRNNLayer(rnn_type, units, dropout, return_state=False, return_sequences=False):
    if rnn_type == 'rnn':
        return SimpleRNN(units, dropout=dropout, return_state=return_state, return_sequences=return_sequences)

    elif rnn_type == 'lstm':
        return LSTM(units, dropout=dropout, return_state=return_state, return_sequences=return_sequences)

    elif rnn_type == 'gru':
        return GRU(units, dropout=dropout, return_state=return_state, return_sequences=return_sequences)

class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.U = tf.keras.layers.Dense(units)
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, enc_state, enc_out):
        enc_state = tf.concat(enc_state, 1)
        enc_state = tf.expand_dims(enc_state, 1)
        print(enc_state.shape, enc_out.shape)
        attention_score = self.V(tf.nn.tanh(self.U(enc_state) + self.W(enc_out)))
        attention_weights = tf.nn.softmax(attention_score, axis=1)
        context_vector = attention_weights * enc_out
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights





class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(ScaledDotProductAttention, self).__init__()
        self.units = units
    # state is the query, key and value come from the encoder output

    def transformer_encoder(self, enc_state, enc_out):
        # Normalization and Attention
        x, attention_score = MultiHeadAttention(
            key_dim=self.units, num_heads=4, dropout=0.2)(enc_state, enc_out, return_attention_scores=True)
        return x, attention_score


    def call(self, enc_state, enc_out):
        enc_state = tf.concat(enc_state, 1)
        enc_state = tf.expand_dims(enc_state, 1)
        context_vector, attention_weights = self.transformer_encoder(enc_state=enc_state, enc_out=enc_out)
        return context_vector, attention_weights





class Encoder(tf.keras.Model):
    def __init__(self, rnn_type, num_layers, units, vocab_size, embedding_dim, dropout):
        super(Encoder, self).__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.units = units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.dropout = dropout
        self.encoder_layer_list = []
        self.initialize_encoder()

    def initialize_encoder(self):
        for _ in range(self.num_layers):
            layer = getRNNLayer(rnn_type=self.rnn_type, units=self.units, dropout=self.dropout, return_sequences=True,
                        return_state=True)
            self.encoder_layer_list.append(layer)

    def call(self, x, hidden):
        x = self.embedding(x)
        x = self.encoder_layer_list[0](x, initial_state=hidden)
        for lstm_layer in self.encoder_layer_list[1:]:
            x = lstm_layer(x)
        output, state = x[0], x[1:]
        return output, state


    def initialize_hidden_state(self, batch_size):
        # https: // tiewkh.github.io / blog / gru - hidden - state /
        if self.rnn_type != "lstm":
            return [tf.zeros((batch_size, self.units))]
        else:
            return [tf.zeros((batch_size, self.units))] * 2



class Decoder(tf.keras.Model):
    def __init__(self, rnn_type, num_layers, units, vocab_size, embedding_dim, dropout, attention_flag=False):
        super(Decoder, self).__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.units = units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.dropout = dropout
        self.decoder_layer_list = []
        self.attention_flag = attention_flag
        if self.attention_flag:
            self.attention_layer = Attention(units=self.units)
        self.initialize_decoder()
        self.dense = Dense(vocab_size, activation="softmax")
        self.flatten = Flatten()


    def initialize_decoder(self):
        for _ in range(self.num_layers):
            layer = getRNNLayer(rnn_type=self.rnn_type, units=self.units, dropout=self.dropout, return_sequences=True,
                        return_state=True)
            self.decoder_layer_list.append(layer)

    def call(self, x, hidden, enc_out=None):
        x = self.embedding(x)
        '''
        Use the context vector(mixture of encoder outputs as the input for the first layer of the decoder)
        '''
        if self.attention_flag:
            context_vector, attention_weights = self.attention_layer(hidden, enc_out)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], -1)
            # print(context_vector.shape, x.shape)
            # x = tf.concat([context_vector, x], -1)
        else:
            attention_weights = None

        x = self.decoder_layer_list[0](x, initial_state=hidden)

        for layer in self.decoder_layer_list[1:]:
            x = layer(x)
        #a tuple of (lstm_output, state_h, state_c) is returned, this is split between output and state
        output, state = x[0], x[1:]
        # print(output.shape)
        #project output of the decoder lstm to match target vocabulary size
        output = self.dense(self.flatten(output))
        #returning attention weights for visualization
        return output, state, attention_weights