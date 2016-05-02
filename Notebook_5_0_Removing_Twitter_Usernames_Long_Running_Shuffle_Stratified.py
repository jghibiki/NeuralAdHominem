
# coding: utf-8

# # Notebook 2: CNN
# 
# ## Overview: 
# 
# 1. Begin by importing and getting the embeddings and word to index mappings we created in [Notebook 1: Embed Words](Notebook_1_Embed_Words.ipynb)
# 

# In[1]:

import cPickle as pickle
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell 
import itertools
from collections import Counter
import csv
import re
import numpy as np
import string


# In[2]:

embeddings = None
mappings = None
rows = None

with open("word_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)
with open("word_mappings.pkl", "rb") as f:
    mappings = pickle.load(f)
    

urlFinder = re.compile('\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*')
atNameFinder = re.compile(r'@([A-Za-z0-9_]+)')
atNameCounter = 0

exclude_punc = set([
        "!",
        "?",
        ".",
        ",",
        ":",
        ";",
        "'",
        "\"",
        "“",
        "’",
        "-"
])

sentences = []
labels = []
x = []
y = []
_y = []

with open('data.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=',')   
    
    for row in reader:
        words = []
        
        for word in row[1]             .strip()             .replace("&amp;", "")             .replace("&gt;","")             .replace("&lt;", "")             .lower().split():
            
            if urlFinder.match(word):
                words.append("<URL/>")
            elif atNameFinder.search(word):
                words.append("<AT_NAME_%s/>" % atNameCounter)
                atNameCounter +=1
            else:
                word = ''.join(ch for ch in word if ch not in exclude_punc)
                words.append(word)
        sentences.append(words)
        labels.append(([0, 1] if row[0] == "example" else [1, 0]))
        _y.append(1 if row[0] == "example" else 0)


sequence_length = max(len(i) for i in sentences)
padded_sentences = []
for i in range(len(sentences)):
    sentence = sentences[i]
    num_padding = sequence_length - len(sentence)
    new_sentence = sentence + ["<PAD/>"] * num_padding
    padded_sentences.append(new_sentence)
    
 
word_counts = Counter(itertools.chain(*padded_sentences))

# Mapping from index to word
vocabulary_inv = [x[0] for x in word_counts.most_common()]
# Mapping from word to index
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

x = np.array([[vocabulary[word] for word in sentence] for sentence in padded_sentences])
y = np.array(labels)


# In[3]:

import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
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


# In[4]:

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


# In[5]:

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from sklearn.cross_validation import StratifiedShuffleSplit

# Parameters
# ==================================================

# Model Hyperparameters
embedding_dim = 128  #Dimensionality of character embedding (default: 128)
filter_sizes =  "3,4,5" #"Comma-separated filter sizes (default: '3,4,5')")
num_filters = 128  #"Number of filters per filter size (default: 128)")
dropout_keep_prob = 0.5 #"Dropout keep probability (default: 0.5)")
l2_reg_lambda = 0.0 #"L2 regularizaion lambda (default: 0.0)")

# Training parameters
batch_size = 64 # "Batch Size (default: 64)")
num_epochs = 200 #"Number of training epochs (default: 200)")
evaluate_every = 100  #"Evaluate model on dev set after this many steps (default: 100)")
checkpoint_every = 100 # "Save model after this many steps (default: 100)")
# Misc Parameters
allow_soft_placement = True # "Allow device soft device placement")
log_device_placement = False  #"Log placement of ops on devices")
display_train_steps = False # toggles output of training step results



# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
# Randomly shuffle data
sss = StratifiedShuffleSplit(_y, 1, test_size=0.5, random_state=0)
for train, test in sss:
    x_train = np.random.permutation(x[train])
    y_train = np.random.permutation(y[train])

    x_dev = np.random.permutation(x[test])
    y_dev = np.random.permutation(y[test])

# Training
# ==================================================


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
            embedding_size=embedding_dim,
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
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "unique-name-replacement", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
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
            zip(x_train, y_train), batch_size, num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
        
        print("\nFinal Evaluations:")
        dev_step(x_dev, y_dev, writer=dev_summary_writer)
  
        
        print("")


# In[6]:

pos_x_dev = []
pos_y_dev = []

neg_x_dev = []
neg_y_dev = []

for y, x in zip(y_dev, x_dev):
    if(y[0] == 0 and y[1] == 1):
        pos_x_dev.append(x)
        pos_y_dev.append(y)
    else:
        neg_x_dev.append(x)
        neg_y_dev.append(y)
even_x_dev = np.array(pos_x_dev + neg_x_dev[:len(pos_x_dev)])
even_y_dev = np.array(pos_y_dev + neg_y_dev[:len(pos_y_dev)])
print(len(even_y_dev), len(even_x_dev))

print("Balanced Evaluation:")
dev_step(even_x_dev, even_y_dev, writer=dev_summary_writer)


# In[7]:

accuracies = []
for y, x in zip(even_y_dev, even_x_dev):
    sent = []
    for word in x:
        sent.append(vocabulary_inv[word])
    print(' '.join(sent))
    print("example" if y[0] == 0 and y[1] == 1 else "nonexample")
    dev_step([x], [y], writer=dev_summary_writer)
    print("")
    print("")
    

for y, x in zip(y_dev, x_dev):
    accuracies.append(dev_step([x], [y], writer=dev_summary_writer))
    
tp = 0
fp = 0
tn = 0
fn = 0

for a, y in zip(accuracies, y_dev):
    if(a == 1.0 and y[0] == 0 and y[1] == 1):
        tp += 1
    elif(a == 0.0 and y[0] == 0 and y[1] == 1):
        fn += 1
    elif(a == 1.0 and y[0] == 1 and y[1] == 0):
        fp += 1
    elif(a == 0.0 and y[0] == 1 and y[1] == 0):
        tn +=1 
        
        
print("True Positives %s" % tp)
print("True Negatives %s" % tn)
print("False Positives %s" % fp)
print("False Negatives %s" % fn)
sensitivity = (tp/(tp+float(fn)))
print("Sensitivity: %s" % sensitivity)
specificity = (tn/(tn+float(fp)))
print("Specificity: %s" % specificity)





# In[ ]:



