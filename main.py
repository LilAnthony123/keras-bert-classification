#! -*- coding:utf-8 -*-
import os
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import numpy as np
from keras_bert_classification.sentiment import data_generator, OurTokenizer
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras_bert_classification.cnews_loader import process_file, read_vocab, read_category
import codecs
from tqdm import tqdm

seq_length = 512
cat_nums = 10
batch_size = 4
epochs = 10

config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
vocab_path = '../chinese_L-12_H-768_A-12/vocab.txt'


base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')

token_dict = {}

with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = OurTokenizer(token_dict)


#vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

words, word_to_id = read_vocab(vocab_path)
categories, cat_to_id = read_category()


#load and process data
x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, seq_length)
#x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, seq_length)
x_train = [x_train, np.zeros_like(x_train)]
x_val = [x_val, np.zeros_like(x_val)]

#fine-tune
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path,training=True, trainable=True, seq_len=seq_length)


#inputs1 = Input(shape=(seq_length,))
#inputs2 = Input(shape=(seq_length,))
#inputs = Input()
inputs = bert_model.input[:2]
#x = Lambda(lambda x: x[:, 0])(x)
x = bert_model.get_layer('NSP-Dense').output
outputs = Dense(units=cat_nums, activation='softmax')(x)
model = Model(inputs, outputs)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(5e-5), # 用足够小的学习率
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint('checkpoint.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

model.summary()

#model.fit([x_train, y_train], batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=[x_val, y_val])
#train_D = data_generator(x_train)
#valid_D = data_generator(x_val)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(x_val,y_val))

'''
model.fit_generator(
    train_D.__iter__(), y_train,
    steps_per_epoch=500,
    epochs=epochs,
    validation_data=valid_D.__iter__(),
    validation_steps=500,
)
'''