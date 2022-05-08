
from assignment_3.ModelFactory import preprocess_data, get_data_files
import wandb
import tensorflow as tf

from assignment_3.Seq2SeqModel import Seq2SeqModel

wandb.login(key='677e9fea45b64f0b222413b502a3fbe87ea3b70e')

config_defaults = {"embedding_dim": 128,
                   "enc_dec_layers": 1,
                   "rnn_type": "lstm",
                   "units": 512,
                   "dropout": 0.1,
                   "attention_flag": True,
                   "teacher_forcing_flag": True
                   }

run = wandb.init(config=config_defaults, project="cs6910-assignment3", entity="adi-rohit")
TRAIN_TSV, VAL_TSV, TEST_TSV = get_data_files("hi")

dataset, input_tokenizer, targ_tokenizer = preprocess_data(TRAIN_TSV)
val_dataset, _, _ = preprocess_data(VAL_TSV, input_tokenizer, targ_tokenizer)

model = Seq2SeqModel(embedding_dim=wandb.config.embedding_dim,
                     num_encoder_layers=wandb.config.enc_dec_layers,
                     num_decoder_layers=wandb.config.enc_dec_layers,
                     rnn_type=wandb.config.rnn_type,
                     units=wandb.config.units,
                     dropout=wandb.config.dropout,
                     attention_flag=wandb.config.attention_flag)

model.set_vocabulary(input_tokenizer, targ_tokenizer)
model.build(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metric=tf.keras.metrics.SparseCategoricalAccuracy())
model.fit(dataset, val_dataset, epochs=30)
