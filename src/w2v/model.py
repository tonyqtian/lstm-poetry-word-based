import tensorflow as tf
import numpy as np
import random
import math
#from tensorflow.models.rnn_cell import rnn


class Model(object):

    def __init__(self, is_training, config, embd=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        self._embedding = embd
        
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device('/cpu:0'):
            if self._embedding == None:
                embedding = tf.get_variable('embedding', [vocab_size, size])
            else:
                embedding = tf.get_variable('embedding', initializer=self.embedding)
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)
        self._embedding = embedding

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        
#         inputs = [tf.squeeze(input_, [1])
#                   for input_ in tf.split(1, num_steps, inputs)]
#         outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self._initial_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable('softmax_w', [size, vocab_size])
        softmax_b = tf.get_variable('softmax_b', [vocab_size])
        self._logits = logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def logits(self):
        return self._logits
    
    @property
    def embedding(self):
        return self._embedding


class Word2Vec(object):
    
    def __init__(self, vocab_size, embed_size):
        self.batch_size = batch_size = 128
        self.vocabulary_size = vocabulary_size = vocab_size
        self.embedding_size = embedding_size = embed_size # Dimension of the embedding vector.
        self.skip_window = 1 # How many words to consider left and right.
        self.num_skips = 2 # How many times to reuse an input to generate a label.
        # We pick a random validation set to sample nearest neighbors. here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent. 
        self.valid_size = valid_size = 16 # Random set of words to evaluate similarity on.
        valid_window = 100 # Only pick dev samples in the head of the distribution.
        self._valid_examples = np.array(random.sample(range(valid_window), valid_size))
        num_sampled = 64 # Number of negative examples to sample.
        
        # Input data.
        self._train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
        self._train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(self._valid_examples, dtype=tf.int32)
        
        # Variables.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        softmax_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                               stddev=1.0 / math.sqrt(embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
        
        # Model.
        # Look up embeddings for inputs.
        embed = tf.nn.embedding_lookup(embeddings, self._train_dataset)
        # Compute the softmax loss, using a sample of the negative labels each time.
        self._loss = tf.reduce_mean(
          tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                     self._train_labels, num_sampled, vocabulary_size))
        
        # Optimizer.
        # Note: The optimizer will optimize the softmax_weights AND the embeddings.
        # This is because the embeddings are defined as a variable quantity and the
        # optimizer's `minimize` method will by default modify all variable quantities 
        # that contribute to the tensor it is passed.
        # See docs on `tf.train.Optimizer.minimize()` for more details.
        self._optimizer = tf.train.AdagradOptimizer(1.0).minimize(self._loss)
        
        # Compute the similarity between minibatch examples and all embeddings.
        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        self._normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(self._normalized_embeddings, valid_dataset)
        self._similarity = tf.matmul(valid_embeddings, tf.transpose(self._normalized_embeddings))
        
    @property
    def train_dataset(self):
        return self._train_dataset
    
    @property
    def train_labels(self):
        return self._train_labels
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def loss(self):
        return self._loss
    
    @property
    def valid_examples(self):
        return self._valid_examples
    
    @property
    def normalized_embeddings(self):
        return self._normalized_embeddings
    
    @property
    def similarity(self):
        return self._similarity