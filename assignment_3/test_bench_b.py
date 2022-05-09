import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
from assignment_3.ModelFactory import preprocess_data, get_data_files
import tensorflow as tf
import pandas as pd
import numpy as np
from assignment_3.Seq2SeqModel import Seq2SeqModel

TRAIN_TSV, VAL_TSV, TEST_TSV = get_data_files("hi")

dataset, input_tokenizer, targ_tokenizer = preprocess_data(TRAIN_TSV)
val_dataset, _, _ = preprocess_data(VAL_TSV, input_tokenizer, targ_tokenizer)
test_dataset, _, _ = preprocess_data(TEST_TSV, input_tokenizer, targ_tokenizer)

model = Seq2SeqModel(embedding_dim=128,
                     num_encoder_layers=1,
                     num_decoder_layers=1,
                     rnn_type="gru",
                     units=512,
                     dropout=0.1,
                     attention_flag=False,
                     teacher_forcing_flag=True)

model.set_vocabulary(input_tokenizer, targ_tokenizer)
model.build(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metric=tf.keras.metrics.SparseCategoricalAccuracy())
model.fit(dataset, val_dataset, epochs=50, log_wandb_flag=False)

test_loss, test_acc = model.evaluate_model(test_dataset, batch_size=100)

test_tsv = pd.read_csv(TEST_TSV, sep="\t", header=None)
inputs = test_tsv[1].astype(str).tolist()
targets = test_tsv[0].astype(str).tolist()

outputs = []

for word in tqdm(inputs):
    outputs.append(model.translate(word, input_tokenizer, targ_tokenizer)[0])
test_acc = np.sum(np.asarray(outputs) == np.array(targets)) / len(outputs)
print("Test Set accuracy: "+str(test_acc))

df = pd.DataFrame()
df["inputs"] = inputs
df["targets"] = targets
df["outputs"] = outputs
df.to_csv("./predictions_vanilla/test_results.csv")
