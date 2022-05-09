import wandb
from Seq2SeqModel import *
wandb.login(key='677e9fea45b64f0b222413b502a3fbe87ea3b70e')

def train_with_wandb(language):
    config_defaults = {"embedding_dim": 128,
                       "enc_dec_layers": 1,
                       "rnn_type": "gru",
                       "units": 512,
                       "dropout": 0.1,
                       "attention_flag": False,
                       "teacher_forcing_flag": True
                       }

    run = wandb.init(config=config_defaults, project="cs6910-assignment3", entity="adi-rohit")

    TRAIN_TSV, VAL_TSV, TEST_TSV = get_data_files(language)

    dataset, input_tokenizer, targ_tokenizer = preprocess_data(TRAIN_TSV)
    val_dataset, _, _ = preprocess_data(VAL_TSV, input_tokenizer, targ_tokenizer)

    model = Seq2SeqModel(embedding_dim=wandb.config.embedding_dim,
                         num_encoder_layers=wandb.config.enc_dec_layers,
                         num_decoder_layers=wandb.config.enc_dec_layers,
                         rnn_type=wandb.config.rnn_type,
                         units=wandb.config.units,
                         dropout=wandb.config.dropout,
                         attention_flag=wandb.config.attention_flag,
                         teacher_forcing_flag=wandb.config.teacher_forcing_flag)

    model.set_vocabulary(input_tokenizer, targ_tokenizer)
    model.build(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metric=tf.keras.metrics.SparseCategoricalAccuracy())
    model.fit(dataset, val_dataset, epochs=30)


# sweep_config = {
#   "method": "grid",
#   "parameters": {
#         "enc_dec_layers": {
#            "values": [1, 2, 3]
#         },
#         "units": {
#             "values": [64, 128, 256, 512]
#         },
#         "rnn_type": {
#             "values": ["gru", "lstm"]
#         },
#         "embedding_dim": {
#             "values": [16]
#         },
#         "enc_dec_layers": {
#             "values": [1, 2, 3]
#         },
#         "dropout": {
#             "values": [0]
#         },
#         "attention_flag": {
#             "values": [False]
#         },
#         "teacher_forcing_flag": {
#             "values": [True, False]
#         }
#     }
# }


sweep_config = {
  "method": "grid",
  "parameters": {
        "embedding_dim": {
            "values": [16, 32, 64, 128]
        },
        "dropout": {
            "values": [0, 0.1, 0.3]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="cs6910-assignment3", entity="adi-rohit")
wandb.agent(sweep_id, project="cs6910-assignment3", function=lambda: train_with_wandb("hi"), entity="adi-rohit")


