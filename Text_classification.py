!pip install -q tf-models-official
!pip install -q -U tensorflow-text
# !pip install -q opencv-python

import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

!wget --no-check-certificate \
    https://storage.googleapis.com/learning-datasets/sarcasm.json \
    -O /tmp/sarcasm.json

import json

with open("/tmp/sarcasm.json", 'r') as f:
    datastore = json.load(f)

datastore[0]

dataset = []
label_dataset = []

for item in datastore:
    dataset.append(item["headline"])
    label_dataset.append(item["is_sarcastic"])

dataset[:10], label_dataset[:10]

import numpy as np

dataset = np.array(dataset)
label_dataset = np.array(label_dataset)

train_size = 0.8
size = int(len(dataset) * train_size)

train_sentence = dataset[:size]
test_sentence = dataset[size:]

train_label = label_dataset[:size]
test_label = label_dataset[size:]

len(train_sentence), len(test_sentence)

import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
tf.io.gfile.listdir(gs_folder_bert)

hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
    lower_case=True)

tokens = tokenizer(tf.constant(["Hello TensorFlow!"]))
tokens

special = tokenizer.get_special_tokens_dict()
special

max_seq_length = 128

packer = tfm.nlp.layers.BertPackInputs(
    seq_length=max_seq_length,
    special_tokens_dict = tokenizer.get_special_tokens_dict())

sentences1 = ["hello tensorflow"]
tok1 = tokenizer(sentences1)
tok1

sentences2 = ["goodbye tensorflow"]
tok2 = tokenizer(sentences2)
tok2

packed = packer([tok1, tok2])

for key, tensor in packed.items():
  print(f"{key:15s}: {tensor[:, :12]}")

class BertInputProcessor(tf.keras.layers.Layer):
  def __init__(self, tokenizer, packer):
    super().__init__()
    self.tokenizer = tokenizer
    self.packer = packer

  def call(self, inputs):
    tok1 = self.tokenizer(inputs['sentence1'])
    tok2 = self.tokenizer(inputs['sentence2'])

    packed = self.packer([tok1, tok2])

    if 'label' in inputs:
      return packed, inputs['label']
    else:
      return packed
    
bert_inputs_processor = BertInputProcessor(tokenizer, packer)

tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])

train_size = 0.8
size = int(len(dataset) * train_size)

train_sentence = dataset[:size]
test_sentence = dataset[size:]

train_label = label_dataset[:size]
test_label = label_dataset[size:]

train_sentence

def encode_sentence(s, tokenizer):
   tokens = list(tokenizer.tokenize(s))
   tokens.append('[SEP]')
   return tokenizer.convert_tokens_to_ids(tokens)

tokenizer(['[CLS]'])

def bert_encode(sentences, tokenizer):
  tokenized_sentences = tf.ragged.constant([
      encode_sentence(s, tokenizer)
      for s in sentences])

  # CLS TOken đứng đầu câu
  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*tokenized_sentences.shape[0]

  input_word_ids = tf.concat([cls, tokenized_sentences], axis=-1)

  input_mask = tf.ones_like(input_word_ids).to_tensor()

  type_cls = tf.zeros_like(cls)

  type_s1 = tf.zeros_like(tokenized_sentences)

  input_type_ids = tf.concat(
      [type_cls, type_s1], axis=-1).to_tensor()

  inputs = {
      'input_word_ids': input_word_ids.to_tensor(),
      'input_mask': input_mask,
      'input_type_ids': input_type_ids}

  return inputs

train_inputs = bert_encode(train_sentence, tokenizer)
train_label_tensors = tf.constant(train_label)

test_inputs = bert_encode(test_sentence, tokenizer)
test_label_tensors = tf.constant(test_label)

import json

bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

bert_config = bert.configs.BertConfig.from_dict(config_dict)

config_dict

bert_classifier, bert_encoder = bert.bert_models.classifier_model(
    bert_config, num_labels=2)

tf.keras.utils.plot_model(bert_classifier, show_shapes=True, dpi=48)

test_batch = {key: val[:10] for key, val in train_inputs.items()}

bert_classifier(
    test_batch, training=True
).numpy()

checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
checkpoint.read(
    os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()

# Set up epochs and steps
epochs = 3
batch_size = 32
eval_batch_size = 32

train_data_size = len(train_label)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

# creates an optimizer with learning rate schedule
optimizer = nlp.optimization.create_optimizer(
    2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)

bert_classifier.fit(
      train_inputs, train_label_tensors,
      batch_size=32,
      validation_data=(test_inputs, test_label_tensors),
      epochs=epochs)