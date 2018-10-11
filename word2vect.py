# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import random
import collections
from collections import Counter
import jieba
from sklearn.manifold import TSNE

'''
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文标签  
mpl.rcParams['font.family'] = 'STSong'
mpl.rcParams['font.size'] = 20
'''

training_file = 'data/test.txt'
    
def get_ch_lable(txt_file):  
    labels= ""
    with open(txt_file, 'rb') as f:
        for label in f: 
            #labels =label.decode('utf-8')
            labels =labels+label.decode('utf8')
    return  labels
   
def fenci(training_data):
    seg_list = jieba.cut(training_data)
    training_ci = " ".join(seg_list)
    training_ci = training_ci.split()
    training_ci = np.array(training_ci)
    training_ci = np.reshape(training_ci, [-1, ])
    return training_ci


def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
 
training_data =get_ch_lable(training_file)
print("总字数",len(training_data))
training_ci =fenci(training_data)
print("总词数",len(training_ci))    
training_label, count, dictionary, words = build_dataset(training_ci, 350)
words_size = len(dictionary)
print("字典词数",words_size)
print('Sample data', training_label[:10], [words[i] for i in training_label[:10]])

data_index = 0
def generate_batch(data,batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span

    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        if data_index == len(data):
            buffer = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(training_label,batch_size=8, num_skips=2, skip_window=1)
for i in range(8):# 取第一个字，后一个是标签，再取其前一个字当标签，
    print(batch[i], words[batch[i]], '->', labels[i, 0], words[labels[i, 0]])


batch_size = 128
embedding_size = 128
skip_window = 1 
num_skips = 2

valid_size = 16 
valid_window =np.int32( words_size/2 )
print("valid_window",valid_window)
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

tf.reset_default_graph()

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

with tf.device('/cpu:0'):
    embeddings = tf.Variable(tf.random_uniform([words_size, embedding_size], -1.0, 1.0))#94个，每个128个向量
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    nce_weights = tf.Variable( tf.truncated_normal([words_size, embedding_size],stddev=1.0 / tf.sqrt(np.float32(embedding_size))))
    nce_biases = tf.Variable(tf.zeros([words_size]))

loss = tf.reduce_mean(
tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                 labels=train_labels, inputs=embed,
                 num_sampled=num_sampled, num_classes=words_size))

optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
print("________________________",similarity.shape)

num_steps = 100001
with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(training_label, batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0
        
        if step % 10000 == 0:
            sim = similarity.eval(session=sess)
            for i in range(valid_size):
                valid_word = words[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]  #argsort函数返回的是数组值从小到大的索引值
                log_str = 'Nearest to %s:' % valid_word

            for k in range(top_k):
                close_word = words[nearest[k]]
                log_str = '%s,%s' % (log_str, close_word)
            print(log_str)
    final_embeddings = normalized_embeddings.eval()
