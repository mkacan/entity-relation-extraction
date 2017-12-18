class Parameters(object):
    def __init__(self):
        with open("parameters.txt", "r") as f:
            lines = filter(lambda x: len(x) > 0, map(lambda l: l.strip(), f.readlines()))

        parameter_dict = {pair[0]: pair[1] for pair in map(lambda x: x.split("="), lines)}

        data_root = parameter_dict["DATA_ROOT"]
        problem_name = parameter_dict["PROBLEM_NAME"]
        batch_size = int(parameter_dict["BATCH_SIZE"])

        self.data_root = data_root
        self.problem_name = problem_name
        self.batch_size = batch_size

        self.data_folder = data_root + problem_name + "/"

        self.SPECIAL_CHAR_UNK = parameter_dict["SPECIAL_CHAR_UNK"]
        self.SPECIAL_CHAR_START = parameter_dict["SPECIAL_CHAR_START"]
        self.SPECIAL_CHAR_END = parameter_dict["SPECIAL_CHAR_END"]

        self.reuse_vocabularies = parameter_dict["REUSE_VOCABULARIES"]
        self.vocabularies_dir = parameter_dict["VOCABULARIES_DIR"]

        self.train_file = data_root + problem_name + "/" + parameter_dict["TRAIN_FILE"]
        self.validate_file = data_root + problem_name + "/" + parameter_dict["VALIDATE_FILE"]
        self.test_file = data_root + problem_name + "/" + parameter_dict["TEST_FILE"]

