# DL-CS6910-Assignment-3

## This folder contains files to run seq2seq character level language translation model.

### The model can be run by using the Seq2SeqModel class present in the Seq2SeqModel.py folder.

Steps for running the model.
1. Set up language tokenizers
```
from assignment_3.ModelFactory import preprocess_data, get_data_files
TRAIN_TSV, VAL_TSV, TEST_TSV = get_data_files("hi")
dataset, input_tokenizer, targ_tokenizer = preprocess_data(TRAIN_TSV)
val_dataset, _, _ = preprocess_data(VAL_TSV, input_tokenizer, targ_tokenizer)
test_dataset, _, _ = preprocess_data(TEST_TSV, input_tokenizer, targ_tokenizer)
```
This code sets up the tokenizers for hindi based translation("hi")

2. Set up the model parameters and run the model

```
model = Seq2SeqModel(embedding_dim=128,
                     num_encoder_layers=1,
                     num_decoder_layers=1,
                     rnn_type="gru",
                     units=512,
                     dropout=0.1,
                     attention_flag=True,
                     teacher_forcing_flag=True)

model.set_vocabulary(input_tokenizer, targ_tokenizer)
model.build(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metric=tf.keras.metrics.SparseCategoricalAccuracy())
model.fit(dataset, val_dataset, epochs=50, log_wandb_flag=False)
```

3. Model Evaluation
```
test_loss, test_acc = model.evaluate_model(test_dataset, batch_size=100)

test_tsv = pd.read_csv(TEST_TSV, sep="\t", header=None)
inputs = test_tsv[1].astype(str).tolist()
targets = test_tsv[0].astype(str).tolist()

outputs = []
print("Translation Started on Test Set....")
for word in tqdm(inputs):
    outputs.append(model.translate(word, input_tokenizer, targ_tokenizer)[0])
test_acc = np.sum(np.asarray(outputs) == np.array(targets)) / len(outputs)
print("Test Set accuracy: " + str(100*test_acc))
```
Please note that the model accuracy on test set is reported on word level and not on the character level. This program also generates csv file containing model predictions. The predictions for the vanilla(attention_flag=False) model is stored under predictions_vanilla/ folder, while attention based model predictions are stored in predictions_attention/ folder.

### Sweeps can be run from Train.py file by setting required hyperparamters as shown below.
```
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
```

