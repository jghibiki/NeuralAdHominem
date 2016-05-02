
# coding: utf-8

# # Notebook 2: CNN
# 
# ## Overview: 
# 
# 1. Begin by importing and getting the embeddings and word to index mappings we created in [Notebook 1: Embed Words](Notebook_1_Embed_Words.ipynb)
# 

# # Word Embedding Parameters

# In[34]:

import numpy as np
import random


embedding_num_steps = 1000001

unknown_word_token = "<UNK/>"
embedding_batch_size = 20
embedding_size = 300 # Dimension of the embedding vector.
skip_window = 10       # How many words to consider left and right.
num_skips = 20         # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(np.arange(valid_window), valid_size))
num_sampled = 64    # Number of negative examples to sample.


# # TextCNN Parameters

# In[35]:

# Model Hyperparameters
sentence_padding_token = "<PAD/>"
sentence_padding = 80

filter_sizes =  "3,4,5" #"Comma-separated filter sizes (default: '3,4,5')")
num_filters = 50  #"Number of filters per filter size (default: 128)")
dropout_keep_prob = 0.5 #"Dropout keep probability (default: 0.5)")
l2_reg_lambda = 3.0 #"L2 regularizaion lambda (default: 0.0)")

# Training parameters
text_cnn_batch_size = 64 # "Batch Size (default: 64)")
num_epochs = 100 #"Number of training epochs (default: 200)")
evaluate_every = 1000  #"Evaluate model on dev set after this many steps (default: 100)")
checkpoint_every = 100000 # "Save model after this many steps (default: 100)")

# Evaluation Parameters
num_folds = 10 # number of cross validation folds 

# Misc Parameters
allow_soft_placement = True # "Allow device soft device placement")
log_device_placement = False  #"Log placement of ops on devices")
display_train_steps = False # toggles output of training step results

run_name = "jumbo-vocab-300-dim-plus-nltk-tokenizer"


# In[20]:

import cPickle as pickle
import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell 
import itertools
from collections import Counter
import csv
import re

import string
import pyprind
import collections
import math
from nltk.tokenize import TweetTokenizer


# # Text Pre-processing Functions

# In[37]:

import re

urlFinder = re.compile('\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*')
atNameFinder = re.compile(r'@([A-Za-z0-9_]+)')

exclude_punc = set([
        "!",
        "?",
        ".",
        ",",
        ":",
        ";",
        "'",
        "\"",
        "'",
        "-",
        "(",
        ")"
])

tknzr = TweetTokenizer()

def clean(string):
    global atNameFinder
    global urlFinder
    global tknzr

    words = []
    
    string = string         .replace("&amp;", "")         .replace("&gt;","")         .replace("&lt;", "")         .lower()
        
    tokens = tknzr.tokenize(string)

    for word in tokens:
        if urlFinder.match(word):
            words.append("<URL/>")
        elif atNameFinder.search(word):
            words.append("<AT_NAME/>")
        else:
            words.append(word)
    return words

def pad(sentence):
    global sentence_padding
    global sentence_padding_token
    if(sentence_padding < len(sentence)):
        raise Exception("Increase sentence_padding,             found sentence that is %s words long. sentence_padding must be             greater than or equal to the number of words in the longest sentence" % len(sentence))
    else:
        for x in range(sentence_padding - len(sentence)):
            sentence.append(sentence_padding_token)
    return sentence


# # Training Word Embeddings

# ## 1. Loading Words

# In[11]:

wordSet = set()
vocabGrowth = 0
vocabulary = {}
vocabulary_inv = []

# Build Vocab
with open('vocab.csv', 'rb') as f:
    
    reader = csv.reader(f, delimiter=',')
    numline = len([row for row in reader])
    bar = pyprind.ProgBar(numline, monitor=True)
    f.seek(0)
    
    for row in reader:
        if len(row) > 0:
            words = clean(row[0])
            for word in words:
                word = word.encode('ascii', 'replace')
                if(word not in wordSet):
                    vocabulary_inv.append(word)
                    vocabulary[word] = vocabulary_inv.index(word)
                    wordSet.add(word)
        bar.update()
                
vocabulary_inv.append(sentence_padding_token)
vocabulary[sentence_padding_token] = vocabulary_inv.index(word)
wordSet.add(sentence_padding_token)

vocabulary_inv.append(unknown_word_token)
vocabulary[unknown_word_token] = vocabulary_inv.index(word)
wordSet.add(unknown_word_token)

vocabulary_size = len(wordSet)
print("Vocabulary Size: %s" % vocabulary_size)

embeddings = None
data_index = 0
data = []


# ## 2. Training Embeddings

# In[28]:

data = [ idx for word, idx in vocabulary.iteritems() ]

print('Sample data %s' % data[:10])

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    global data
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
        while target in targets_to_avoid:
            target = random.randint(0, span - 1)
        targets_to_avoid.append(target)
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=10, num_skips=10, skip_window=5)

for i in range(10):
    print('%s -> %s' % (batch[i], labels[i, 0]))
    print('%s -> %s' % (vocabulary_inv[batch[i]], vocabulary_inv[labels[i, 0]]))



graph = tf.Graph()
with graph.as_default():
    # Input da 4ta.
    train_inputs = tf.placeholder(tf.int32, shape=[embedding_batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[embedding_batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # Construct the variables for the NCE loss
        with tf.name_scope("nce_weights") as scope:
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        nce_biases_hist = tf.histogram_summary("nce_biases", nce_biases)

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    with tf.name_scope("loss") as scope:
        loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                         num_sampled, vocabulary_size))
    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope("train") as scope:
        optimizer = tf.train.GradientDescentOptimizer(0.25).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
    valid_embeddings, normalized_embeddings, transpose_b=True)


    # Step 5: Begin training.

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/tensor_logs/expiriment_1", session.graph_def)

        #Adds an op to initialize all variables
        init_op = tf.initialize_all_variables()

        # Begins running the init opp
        init_op.run()

        print("Initialized")
        average_loss = 0
        bar = pyprind.ProgBar(embedding_num_steps, monitor=True)
        for step in xrange(embedding_num_steps):
            batch_inputs, batch_labels = generate_batch(
                embedding_batch_size, num_skips, skip_window)
            feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            summary_str, _, loss_val = session.run([merged, optimizer, loss], feed_dict=feed_dict)
            writer.add_summary(summary_str, step)
            average_loss += loss_val
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print("Average loss at step %s: %s" % (step, average_loss))
                average_loss = 0
            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 5000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = vocabulary_inv[valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = vocabulary_inv[nearest[k]]
                    log_str = "%s %s" % (log_str, close_word)
                print(log_str)
            bar.update()

        # eval embedding tensor
        embeddings = normalized_embeddings.eval()


# # Training TextCNN Model

# In[57]:

sentences = []
labels = []
_y = []
with open('data.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=',')   
    
    for row in reader:
        sentences.append(clean(row[1]))
        labels.append(([0, 1] if row[0] == "example" else [1, 0]))
        _y.append(1 if row[0] == "example" else 0)


sequence_length = max(len(i) for i in sentences)
padded_sentences = [ pad(sentence) for sentence in sentences]
    
 
word_counts = Counter(itertools.chain(*padded_sentences))

# Mapping from index to word
vocabulary_inv = [x[0] for x in word_counts.most_common()]
# Mapping from word to index
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

x = np.array([[vocabulary[word] for word in sentence] for sentence in padded_sentences])
y = np.array(labels)
_y = np.array(_y)


# In[58]:

import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, embedding_tensor, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(embedding_tensor,
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# In[60]:

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# In[62]:

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from sklearn.cross_validation import StratifiedKFold
import sys




# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
sys.stdout.flush()
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
_y_shuffled = _y[shuffle_indices]

skf = StratifiedKFold(_y_shuffled, n_folds=num_folds)

fold_accuracies = []
fold_specificities = []
fold_sensitivities = []

# Split train/test set
# TODO: This is very crude, should use cross-validation
#x_train, x_dev = x_shuffled[:len(x_shuffled)-1], x_shuffled[-len(x_shuffled)-1:]
#y_train, y_dev = y_shuffled[:len(y_shuffled)-1], y_shuffled[-len(y_shuffled)-1:]
#print("Vocabulary Size: {:d}".format(len(vocabulary)))
#print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
#print("Train Pos/Dev Pos Split {:d}/{:d}"
#      .format(
#        len(
#            [y for y in y_train if y[0] == 0 and y[1] == 1]
#        ), len(
#            [y for y in y_dev if y[0] == 0 and y[1] == 1]
#        )))
#print("Train Neg/Dev Neg Split {:d}/{:d}"
#      .format(
#        len(
#            [y for y in y_train if y[0] == 1 and y[1] == 0]
#        ), len(
#            [y for y in y_dev if y[0] == 1 and y[1] == 0]
#        )))


# Training
# ==================================================

foldBar = pyprind.ProgBar(num_folds, title='CV_Progress')
for idx, fold in zip(skf, range(num_folds)):
    x_train = x[idx[0]]
    y_train = y[idx[0]]
    
    x_dev = x[idx[1]]
    y_dev = y[idx[1]]
    
    
    
    
    
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=allow_soft_placement,
          log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=2,
                vocab_size=len(vocabulary),
                embedding_size=embedding_size,
                embedding_tensor=embeddings,
                filter_sizes=map(int, filter_sizes.split(",")),
                num_filters=num_filters,
                l2_reg_lambda=l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", run_name, timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train", str(fold))
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev", str(fold))
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())
    
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if(display_train_steps):
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                
                if writer:
                    writer.add_summary(summaries, step)
                return accuracy

            # Generate batches
            batches = batch_iter(
                zip(x_train, y_train), text_cnn_batch_size, num_epochs)
            # Training loop. For each batch...
            it = ((int(len(data)/text_cnn_batch_size)+1)*num_epochs)
            print("Steps: {0}".format(it))
            bar = pyprind.ProgBar(it, title='fold_{0}'.format(fold), monitor=True)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                
                bar.update()
                sys.stderr.flush()
            
            acc = dev_step(x_dev, y_dev, writer=dev_summary_writer)
            fold_accuracies.append(acc)
            
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            
            for _y, _x in zip(y_dev, x_dev):
                a= dev_step([_x], [_y])
                expected = "example" if _y[0] == 0 and _y[1] == 1  else "nonexample"
                actual = None
                if(_y[0] == 0 and _y[1] == 1):
                    # correct label is example
                    if(a == 1.0):
                        actual = "example"
                    else:
                        actual = "nonexample"
                elif(_y[0] == 1 and _y[1] == 0):
                    if(a == 1.0):
                        actual = "nonexample"
                    else:
                        actual = "example"

                if(expected == "example" and actual == "example"):
                    tp += 1
                elif(expected == "example" and actual == "nonexample"):
                    fn += 1
                elif(expected == "nonexample" and actual =="exaple"):
                    fp += 1
                elif(expected == "nonexample" and actual == "nonexample"):
                    tn +=1 

            sensitivity = (tp/(tp+float(fn)))
            fold_sensitivities.append(sensitivity)

            specificity = (tn/(tn+float(fp)))
            fold_specificities.append(specificity)
            foldBar.update()
            
            sys.stderr.flush()

            
  
        
        print("")
        


# In[7]:

final_accuracy = sum(fold_accuracies) / float(len(fold_accuracies))
print("10-fold final accuracy: %s" % final_accuracy)
final_specificity = sum(fold_specificities) / float(len(fold_specificities))
print("10-fold final specificity: %s" % final_specificity)
final_sensitivities = sum(fold_sensitivities) / float(len(fold_sensitivities))
print("10-fold final sensitivity: %s" % final_sensitivities)


# In[ ]:




# In[ ]:



