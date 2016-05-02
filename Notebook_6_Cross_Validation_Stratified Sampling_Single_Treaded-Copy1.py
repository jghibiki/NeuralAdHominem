
# coding: utf-8

# # Notebook 2: CNN
# 
# ## Overview: 
# 
# 1. Begin by importing and getting the embeddings and word to index mappings we created in [Notebook 1: Embed Words](Notebook_1_Embed_Words.ipynb)
# 

# In[1]:

# Parameters
# ==================================================

# Model Hyperparameters
embedding_dim = 20  #Dimensionality of character embedding (default: 128)
filter_sizes =  "3,4,5" #"Comma-separated filter sizes (default: '3,4,5')")
num_filters = 50  #"Number of filters per filter size (default: 128)")
dropout_keep_prob = 0.5 #"Dropout keep probability (default: 0.5)")
l2_reg_lambda = 3.0 #"L2 regularizaion lambda (default: 0.0)")

# Training parameters
batch_size = 64 # "Batch Size (default: 64)")
num_epochs = 500 #"Number of training epochs (default: 200)")
evaluate_every = 100  #"Evaluate model on dev set after this many steps (default: 100)")
checkpoint_every = 10000 # "Save model after this many steps (default: 100)")

# Evaluation Parameters
num_folds = 10 # number of cross validation folds 

# Misc Parameters
allow_soft_placement = True # "Allow device soft device placement")
log_device_placement = False  #"Log placement of ops on devices")
display_train_steps = False # toggles output of training step results

run_name = "10f-cv-unique-name-replacement"


# In[2]:

import cPickle as pickle
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell 
import itertools
from collections import Counter
import csv
import re
import numpy as np
import string


# In[3]:

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
_y = np.array(_y)


# In[4]:

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


# In[5]:

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


# In[6]:

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from sklearn.cross_validation import StratifiedKFold




# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
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
                zip(x_train, y_train), batch_size, num_epochs)
            # Training loop. For each batch...
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
            
            print("\nFinal Evaluation for fold %s:" % fold)
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

                print("")

            print("True Positives %s" % tp)
            print("True Negatives %s" % tn)
            print("False Positives %s" % fp)
            print("False Negatives %s" % fn)
            sensitivity = (tp/(tp+float(fn)))
            fold_sensitivities.append(sensitivity)
            print("Sensitivity: %s" % sensitivity)
            specificity = (tn/(tn+float(fp)))
            fold_specificities.append(specificity)
            print("Specificity: %s" % specificity)
  
        
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



