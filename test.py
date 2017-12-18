import tensorflow as tf

with tf.variable_scope("TRAIN"):
    v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
    a = tf.constant(2, name="a")
    b = tf.constant(3, name="b")

    c = tf.add(a, b, name="c") * v

with tf.variable_scope("TEST"):
    v_t = tf.get_variable("v", shape=(), initializer=tf.ones_initializer())
    a_t = tf.constant(2, name="a")
    b_t = tf.constant(3, name="b")

    c_t = tf.add(a, b, name="c") * v_t


writer = tf.summary.FileWriter(logdir="logs/")

with tf.Session() as sess:
    writer.add_graph(sess.graph)
    writer.flush()
    writer.close()
