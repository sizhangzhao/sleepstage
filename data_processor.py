import findspark
import pyspark
from pyspark.sql import SparkSession
import mne
from mne.io import read_raw_edf
import os
import itertools
from utils import *
import pickle
import string
import math

CURR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_PATH, "sleep-edf-database-expanded-1.0.0")
EDF_DATA_DIR = os.path.join(DATA_DIR, "sleep-cassette")


class DataProcessor:
    """
    You need to be very careful to run this process, some of the raw data file can't be processed,
    you should add them to removed_subject list to let the process run successfully
    The script has the ability to detect if transformation for a single subject is done,
    so you don't need to worry if it fails for one single subject 
    """

    def __init__(self):
        # findspark.init()
        # self.sc = pyspark.SparkContext(appName="sleep")
        self.removed_subject = {}# {36, 52, 13, 78, 79, 68, 69, 39, 20, 72, 71, 73, 64, 33}
        self.subjects = {i for i in range(66, 83)} #83
        self.chosen_subjects = self.subjects.difference(self.removed_subject)
        self.nights = {1, 2}
        self.file_name_psg = "SC4{0}{1}{2}0-PSG.edf"
        self.file_name_hyp = "SC4{0}{1}{2}{3}-Hypnogram.edf"
        self.file_name_summary = "SC-subjects.xls"
        self.mapping = {'EOG horizontal': 'eog',
                        'Resp oro-nasal': 'misc',
                        'EMG submental': 'misc',
                        'Temp rectal': 'misc',
                        'Event marker': 'misc',
                        'EEG Fpz-Cz': "eeg",
                        'EEG Pz-Oz': "eeg"}
        self.chosen_channel = ["EEG Fpz-Cz", "EEG Pz-Oz"]
        self.frequency_names = ["delta", "theta", "alpha", "sigma", "beta"]
        self.frequency_feature_names = self.get_frequency_feature_names()
        self.annotation_desc_2_event_id = {'Sleep stage W': 1,
                                           'Sleep stage 1': 2,
                                           'Sleep stage 2': 3,
                                           'Sleep stage 3': 4,
                                           'Sleep stage 4': 4,
                                           'Sleep stage R': 5}
        self.event_id = {'Sleep stage W': 1,
                         'Sleep stage 1': 2,
                         'Sleep stage 2': 3,
                         'Sleep stage 3/4': 4,
                         'Sleep stage R': 5}
        self.split_number = 300

    def get_frequency_feature_names(self):
        catisian_list = itertools.product(self.frequency_names, self.chosen_channel)
        frequency_feature_names = [a[1] + "_" + a[0] for a in catisian_list]
        return frequency_feature_names

    def process_single_file(self, patient_id, night_id):
        """
        This function is to process one single file so raw df, label, time domain and frequency domain
        features will all be saved
        """
        print("Processing patient {} and night {}".format(patient_id, night_id))
        file_psg = self.find_e_file_name(patient_id, night_id)
        file_h = self.find_h_file_name(patient_id, night_id)
        data_psg = read_raw_edf(file_psg)
        data_h = mne.read_annotations(file_h)
        data_psg.set_annotations(data_h, emit_warning=False)
        data_psg.set_channel_types(self.mapping)
        events_train, _ = mne.events_from_annotations(
            data_psg, event_id=self.annotation_desc_2_event_id, chunk_duration=30.)
        tmax = 30. - 1. / data_psg.info['sfreq']  # tmax in included
        epochs = mne.Epochs(raw=data_psg, events=events_train,
                            event_id=self.event_id, tmin=0., tmax=tmax, baseline=None)
        relative_power_band = eeg_power_band(epochs)
        freq_features = relative_power_band
        freq_feature_df = pd.DataFrame(freq_features, columns=self.frequency_feature_names)
        epoch_df, time_feature_df, labels = get_series_and_features(epochs, self.chosen_channel)
        features = pd.concat([freq_feature_df, time_feature_df], axis=1)
        res = {"label": labels, "feature": features, "eeg": epoch_df}
        file_name = "data/" + patient_id + night_id + ".pkl"
        return res, file_name

    def save_file(self, file_name, res):
        with open(file_name, 'wb') as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def load_file(self, patient_id, night_id):
        print("Loading patient {} and night {}".format(patient_id, night_id))
        file_name = "data/" + patient_id + night_id + ".pkl"
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        return data

    def split_data_to_batch(self, data, split_number):
        """
        This is to split epochs into bucket, so they are ready for RNN
        This is to split dataset for hand made RNN model
        """
        labels = data["label"]
        features = data["feature"]
        length = len(labels)
        num_batch = math.ceil(length / split_number)
        batches = []
        for i in range(num_batch):
            batch_label = labels[i * split_number: (i+1) * split_number]
            batch_feature = features.iloc[i * split_number: (i+1) * split_number, :]
            batch_feature.reset_index(inplace=True, drop=True)
            batches.append((batch_feature, batch_label))
        return batches

    def split_eeg_data_to_batch(self, data, split_number):
        """
        This is to split epochs into bucket, so they are ready for RNN
        This is to split dataset for CNN RNN model
        """
        labels = data["label"]
        features = data["eeg"]
        length = len(labels)
        num_batch = math.ceil(length / split_number)
        batches = []
        for i in range(num_batch):
            batch_label = labels[i * split_number: (i + 1) * split_number]
            batch_feature = features[i * split_number: (i + 1) * split_number]
            batches.append((batch_feature, batch_label))
        return batches

    def is_data_file_missing(self, patient_id, night_id):
        """
        This is to check if we have a processed data for one subject on one night
        """
        file_name = "data/" + patient_id + night_id + ".pkl"
        return not os.path.exists(file_name)

    def get_label(self, res):
        return res["label"]

    def find_e_file_name(self, patient_id, night_id):
        """
        to find the right name for PSG file name for a given subject and night
        """
        chars = string.ascii_uppercase
        for first_char in chars:
            file_name = self.file_name_psg.format(patient_id, night_id, first_char)
            if os.path.exists(os.path.join(EDF_DATA_DIR, file_name)):
                return os.path.join(os.path.join(EDF_DATA_DIR, file_name))
        return

    def find_h_file_name(self, patient_id, night_id):
        """
        to find the right name for Hypnogram file name for a given subject and night
        """
        chars = string.ascii_uppercase
        for first_char in chars:
            for second_char in chars:
                file_name = self.file_name_hyp.format(patient_id, night_id, first_char, second_char)
                if os.path.exists(os.path.join(EDF_DATA_DIR, file_name)):
                    return os.path.join(os.path.join(EDF_DATA_DIR, file_name))
        return

    def save_file_spark(self):
        """
        spark entry point to process raw file and save the results to pickle files
        """
        chosen_subjects = [str(a) if a > 9 else "0" + str(a) for a in self.chosen_subjects]
        nights = [str(a) for a in self.nights]
        file_combinations = [a for a in itertools.product(chosen_subjects, nights)]

        findspark.init()
        spark_session = SparkSession.builder.appName("dataprocessor").getOrCreate()
        sc = spark_session.sparkContext
        subject_rdd = sc.parallelize(file_combinations)
        subject_rdd = subject_rdd.filter(lambda subject: self.is_data_file_missing(subject[0], subject[1]))
        subject_rdd = subject_rdd.map(lambda subject: self.process_single_file(subject[0], subject[1]))
        subject_rdd.map(lambda subject_res: self.save_file(subject_res[1], subject_res[0])).collect()
        # subject_rdd.map(lambda subject: self.find_e_file_name(subject[0], subject[1])).map(
        #     lambda file: print(file)).collect()
        return

    def load_file_spark(self):
        """
        main function to split data into buckets for hand made rnn model
        the data used for the model is the output of this function
        In this function, you need to change self.chosen_subjects to specify which subject will be saved
        in the file, i.e. which subjects should go into train, val and test
        so three calls are needed
        """
        chosen_subjects = [str(a) if a > 9 else "0" + str(a) for a in self.chosen_subjects]
        nights = [str(a) for a in self.nights]
        file_combinations = [a for a in itertools.product(chosen_subjects, nights)]

        findspark.init()
        spark_session = SparkSession.builder.config("spark.driver.memory", "18g").appName("dataprocessor").getOrCreate()
        sc = spark_session.sparkContext
        subject_rdd = sc.parallelize(file_combinations)
        subject_rdd = subject_rdd.filter(lambda subject: not self.is_data_file_missing(subject[0], subject[1]))
        subject_rdd = subject_rdd.map(lambda subject: self.load_file(subject[0], subject[1]))
        subject_rdd.cache()
        # labels = subject_rdd.map(lambda subject: self.get_label(subject)).reduce(lambda label1, label2: np.concatenate((label1, label2)))
        batches = subject_rdd.map(lambda subject: {"label": subject["label"], "feature": subject["feature"]}).map(lambda subject: self.split_data_to_batch(subject, 50)).reduce(lambda batch1, batch2: batch1 + batch2)
        with open("data/test_rnn.pkl", 'wb') as f:
            pickle.dump(batches, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def load_eeg_file_spark(self):
        """
        main function to split data into buckets for cnn rnn model
        the data used for the model is the output of this function
        you have to similarly specify subject id in line one to split them into train, test and val
        so three calls are needed
        """
        chosen_subjects = [str(a) if a > 9 else "0" + str(a) for a in [i for i in range(10, 12)]] # 8, 10
        nights = [str(a) for a in self.nights]
        file_combinations = [a for a in itertools.product(chosen_subjects, nights)]

        findspark.init()
        spark_session = SparkSession.builder.config("spark.driver.memory", "18g").config("spark.driver.maxResultSize", "10g").appName("dataprocessor").getOrCreate()
        sc = spark_session.sparkContext
        subject_rdd = sc.parallelize(file_combinations)
        subject_rdd = subject_rdd.filter(lambda subject: not self.is_data_file_missing(subject[0], subject[1]))
        subject_rdd = subject_rdd.map(lambda subject: self.load_file(subject[0], subject[1]))
        subject_rdd.cache()
        # labels = subject_rdd.map(lambda subject: self.get_label(subject)).reduce(lambda label1, label2: np.concatenate((label1, label2)))
        batches = subject_rdd.map(lambda subject: {"label": subject["label"], "eeg": subject["eeg"]}).map(lambda subject: self.split_eeg_data_to_batch(subject, 50)).reduce(lambda batch1, batch2: batch1 + batch2)
        with open("data/test_cnn_rnn.pkl", 'wb') as f:
            pickle.dump(batches, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def load_training_data(self):
        with open("data/train.pkl", 'rb') as f:
            data = pickle.load(f)
        return data

    def load_test_eeg_data(self):
        with open("data/test_cnn_rnn.pkl", 'rb') as f:
            data = pickle.load(f)
        return data

if __name__ == "__main__":
    processor = DataProcessor()
    # processor.process_single_file("19", "1")
    # processor.save_file_spark()
    # processor.find_e_file_name("19", "1")
    # data = processor.load_file("05", "1")
    # processor.split_data_to_batch(data)
    processor.load_file_spark()
    # processor.load_training_data()
    # data = processor.load_file("01", "1")
    # processor.split_eeg_data_to_batch(data, 50)
    # processor.load_eeg_file_spark()
    # processor.load_test_eeg_data()
