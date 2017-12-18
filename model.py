import tensorflow as tf

class Model(object):
    def __init__(self, parameters, dataset):
        self.parameters = parameters
        self.dataset = dataset

        self._build_graph()

    def _build_graph(self):
        graph = tf.Graph()

        with graph.as_default() as g:
            self._construct_model()

        self.graph = graph

    def _construct_model(self):
        BATCH_SIZE = self.parameters.batch_size

        """
        ------------------------------- EMBEDDING LAYER START -------------------------------
        """


        # 1. Create placeholders for the word list, POS tag list, label list.
        self.word_sequence = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, None], name="word_sequence")
        self.POS_tag_sequence = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, None], name="POS_tag_sequence")

        self.label_sequence = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, None], name="label_sequence")


        # 2. Embed the words and the POS tags, each with its own learnable embedding matrix.


        # 3. Concatenate the word and POS embeddings into a list of vectors.


        """
        ------------------------------- EMBEDDING LAYER END -------------------------------
        """

        """
        ------------------------------- SEQUENCE LAYER START -------------------------------
        """

        # 4. Run a BiLSTM on the concatenated embeddings.

        # 5. Run a custom RNN that predicts the entity labels:
        #       - input: the output vector of the previous BiLSTM layer
        #       - hidden state:
        #           - incoming: the predicted entity label for the previous input
        #           - outgoing: the predicted entity label for the current input
        #       - output: the logits on the vocabulary of labels (NE tags); unnormalized log-probability of each label for the current input


        """
        ------------------------------- SEQUENCE LAYER END -------------------------------
        """




