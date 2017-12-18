import numpy as np
import _pickle
import os

class Dataset(object):
    def __init__(self, parameters):
        #self.word_to_index, self.index_to_word = self.create_vocabulary(parameters.train_file)

        self.phase_file_dict = {
            "train": parameters.train_file,
            "validate": parameters.validate_file,
            "test": parameters.test_file
        }

        self.current_file = None
        self.phase = None

        self.parameters = parameters

        self._create_vocabularies(parameters.train_file)


    def next_batch(self):
        word_lists = list()
        pos_lists = list()
        label_lists = list()

        batch_sents = list()
        for _ in range(self.parameters.batch_size):
            if self.phase is None:
                break

            batch_sents.append(self.next_sentence())

        if len(batch_sents) < self.parameters.batch_size:
            return None

        for word_list, pos_list, label_list in batch_sents:
            word_lists.append(word_list)
            pos_lists.append(pos_list)
            label_lists.append(label_list)

        return word_lists, pos_lists, label_lists

    def next_batch_np(self):
        word_lists, pos_lists, label_lists = self.next_batch()

        max_len = len(max(word_lists, key=lambda x: len(x)))

        batch_size = self.parameters.batch_size

        words_np = np.zeros(shape=(batch_size, max_len))
        POSs_np = np.zeros(shape=(batch_size, max_len))
        labels_np = np.zeros(shape=(batch_size, max_len))
        mask_np = np.zeros(shape=(batch_size, max_len))

        word_UNK_index = self.word_to_index[self.parameters.SPECIAL_CHAR_UNK]
        POS_tag_UNK_index = self.POS_tag_to_index[self.parameters.SPECIAL_CHAR_UNK]
        NE_tag_UNK_index = self.NE_tag_to_index[self.parameters.SPECIAL_CHAR_UNK]

        for i, (word_list, POS_tag_list, NE_tag_list) in enumerate(zip(word_lists, pos_lists, label_lists)):
            word_index_list = list(map(lambda x: self.word_to_index[x] if x in self.word_to_index else word_UNK_index, word_list))
            POS_tag_index_list = list(map(lambda x: self.POS_tag_to_index[x] if x in self.POS_tag_to_index else POS_tag_UNK_index, POS_tag_list))
            NE_tag_index_list = list(map(lambda x: self.NE_tag_to_index[x] if x in self.NE_tag_to_index else NE_tag_UNK_index, NE_tag_list))

            length = len(word_index_list)

            words_np[i, :length] = word_index_list
            POSs_np[i, :length] = POS_tag_index_list
            labels_np[i, :length] = NE_tag_index_list
            mask_np[i, :length] += 1

        return words_np, POSs_np, labels_np, mask_np

    def next_sentence(self):
        word_list = list()
        pos_list = list()
        label_list = list()

        line = self.current_file.readline()
        if line == "":
            self.end_epoch()
            return None

        word, pos, label = self.parse_word_line(line)

        word_list.append(word)
        pos_list.append(pos)
        label_list.append(label)

        last_pos = self.current_file.tell()
        line = self.current_file.readline()

        while True:
            if line.startswith("Sentence: "):
                self.current_file.seek(last_pos)
                break
            elif line == "":
                self.end_epoch()
                break

            word, pos, label = self.parse_word_line(line)

            word_list.append(word)
            pos_list.append(pos)
            label_list.append(label)

            last_pos = self.current_file.tell()
            line = self.current_file.readline()

        return word_list, pos_list, label_list


    def start_epoch(self, phase):
        if self.current_file is not None:
            self.current_file.close()

        self.current_file = open(self.phase_file_dict[phase])
        self.phase = phase

    def end_epoch(self):
        self.current_file.close()
        self.current_file = None
        self.phase = None


    def _create_vocabularies(self, train_sentences):
        if self.parameters.reuse_vocabularies and os.path.isfile(self.parameters.vocabularies_dir + "vocabularies_dict.pickle"):
            self._load_cached_vocabularies()
            return

        self.start_epoch("train")

        word_set = set()
        POS_tag_set = set()
        NE_tag_set = set()

        while self.phase == "train":
            words, POS_tags, NE_tags = self.next_sentence()

            word_set.update(words)
            POS_tag_set.update(POS_tags)
            NE_tag_set.update(NE_tags)

        # add the special token UNK for all words/POS_tags/NE_tags not seen in the training set
        word_set.add(self.parameters.SPECIAL_CHAR_UNK)
        POS_tag_set.add(self.parameters.SPECIAL_CHAR_UNK)
        NE_tag_set.add(self.parameters.SPECIAL_CHAR_UNK)

        index_to_word, word_to_index = self.create_bidirectional_dicts(word_set)
        index_to_POS_tag, POS_tag_to_index = self.create_bidirectional_dicts(POS_tag_set)
        index_to_NE_tag, NE_tag_to_index = self.create_bidirectional_dicts(NE_tag_set)

        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.POS_tag_to_index = POS_tag_to_index
        self.index_to_POS_tag = index_to_POS_tag
        self.NE_tag_to_index = NE_tag_to_index
        self.index_to_NE_tag = index_to_NE_tag

        vocabularies_dict = dict()
        vocabularies_dict["word_to_index"] = word_to_index
        vocabularies_dict["index_to_word"] = index_to_word
        vocabularies_dict["POS_tag_to_index"] = POS_tag_to_index
        vocabularies_dict["index_to_POS_tag"] = index_to_POS_tag
        vocabularies_dict["NE_tag_to_index"] = NE_tag_to_index
        vocabularies_dict["index_to_NE_tag"] = index_to_NE_tag

        with open(self.parameters.vocabularies_dir + "vocabularies_dict.pickle", "wb") as f:
            _pickle.dump(vocabularies_dict, f, 2)


    def create_bidirectional_dicts(self, element_set):
        enumerated_list = list(enumerate(element_set))

        element_to_index = {element: index for index, element in enumerated_list}
        index_to_element = {index: element for index, element in enumerated_list}

        return index_to_element, element_to_index

    def parse_word_line(self, line):

        if '"' in line:
            line = line.replace('""""', '<$quot$>')
            parts = line.split('"')

            new_parts = list()

            odd_n = True
            for part in parts:
                if odd_n:
                    for p in part.split(","):
                        new_parts.append(p)
                else:
                    new_parts.append(part)
                odd_n = not odd_n

            parsed = list(filter(lambda x: len(x) > 0, new_parts))
            parsed = list(map(lambda x: '"' if x == '<$quot$>' else x, parsed))

            [word, pos, label] = parsed
        else:
            splitted = line.split(",")
            [sent, word, pos, label] = splitted

        return word.strip(), pos.strip(), label.strip()

    def _load_cached_vocabularies(self):
        dir_path = self.parameters.vocabularies_dir

        with open(dir_path + "vocabularies_dict.pickle", "rb") as f:
            vocabularies_dict = _pickle.load(f)

        self.word_to_index = vocabularies_dict["word_to_index"]
        self.index_to_word = vocabularies_dict["index_to_word"]
        self.POS_tag_to_index = vocabularies_dict["POS_tag_to_index"]
        self.index_to_POS_tag = vocabularies_dict["index_to_POS_tag"]
        self.NE_tag_to_index = vocabularies_dict["NE_tag_to_index"]
        self.index_to_NE_tag = vocabularies_dict["index_to_NE_tag"]
