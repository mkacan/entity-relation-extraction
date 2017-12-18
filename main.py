from parameters import Parameters
from dataset import Dataset
from model import Model
from logger import Logger
from trainer import Trainer
import tensorflow as tf

if __name__ == '__main__':


    parameters = Parameters()

    dataset = Dataset(parameters)

    dataset.start_epoch("train")

    model = Model(parameters, dataset)

    with tf.Session(graph=model.graph) as sess:
        sess.run(tf.global_variables_initializer())

        word_indices, POS_tag_indices, NE_tag_indices, mask_indices = dataset.next_batch_np()

        print(word_indices)

        feed_dict = {
            model.word_sequence: word_indices,
            model.POS_tag_sequence: POS_tag_indices,
            model.label_sequence: NE_tag_indices
        }


        [w_seq] = sess.run(fetches=[model.word_sequence], feed_dict=feed_dict)

    print(w_seq)
    exit()

    logger = Logger(parameters)

    trainer = Trainer(parameters)

    trainer.train(dataset, model, logger)
