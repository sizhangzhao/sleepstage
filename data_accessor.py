import pickle
import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import findspark
import pyspark
from pyspark.sql import SparkSession


class DataAccessor:

    def __init__(self):
        self.rnn_only_train = "data/train_rnn.pkl"
        self.rnn_only_val = "data/val_rnn.pkl"
        self.rnn_only_test = "data/test_rnn.pkl"
        self.cnn_rnn_train = "data/train_cnn_rnn.pkl"
        self.cnn_rnn_val = "data/val_cnn_rnn.pkl"
        self.cnn_rnn_test = "data/test_cnn_rnn.pkl"

    def load_pickle(self, file_name):
        """
        general function to load pickle file so we can load individual data
        """
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        return data

    # the below 6 functions are just used to load the corresponding data
    def load_rnn_only_train(self):
        return self.load_pickle(self.rnn_only_train)

    def load_rnn_only_val(self):
        return self.load_pickle(self.rnn_only_val)

    def load_rnn_only_test(self):
        return self.load_pickle(self.rnn_only_test)

    def load_cnn_rnn_train(self):
        return self.load_pickle(self.cnn_rnn_train)

    def load_cnn_rnn_val(self):
        return self.load_pickle(self.cnn_rnn_val)

    def load_cnn_rnn_test(self):
        return self.load_pickle(self.cnn_rnn_test)

    def single_distribution(self, data):
        """
        function to check class distribution among dataset
        """
        data = [batch_data[1] for batch_data in data]
        data = reduce((lambda x1, x2: np.concatenate((x1, x2))), data)
        data = data.astype(np.int64)
        all_sum = len(data)
        bin_sum = np.bincount(data)
        return bin_sum, all_sum, bin_sum / all_sum

    def get_rnn_only_weight(self):
        """
        get counts for individual class in hand made rnn model
        this is used for weighted sample in the training process
        """
        data = self.load_rnn_only_train()
        _, _, weights = self.single_distribution(data)
        return weights[1:]

    def get_cnn_rnn_weight(self):
        """
        get counts for individual class in cnn rnn model
        this is used for weighted sample in the training process
        """
        data = self.load_cnn_rnn_train()
        _, _, weights = self.single_distribution(data)
        return weights[1:]

    def rnn_only_distribution(self):
        """
        get distribution among train, val and test set in hand made rnn model
        """
        data = self.load_rnn_only_train()
        bin_sum, all_sum, weights = self.single_distribution(data)
        print("For training data for rnn only, bin sum is {}, total sum is {}, weights are {}".format(bin_sum, all_sum,
                                                                                                      weights))
        data = self.load_rnn_only_val()
        bin_sum, all_sum, weights = self.single_distribution(data)
        print(
            "For validation data for rnn only, bin sum is {}, total sum is {}, weights are {}".format(bin_sum, all_sum,
                                                                                                      weights))
        data = self.load_rnn_only_test()
        bin_sum, all_sum, weights = self.single_distribution(data)
        print("For test data for rnn only, bin sum is {}, total sum is {}, weights are {}".format(bin_sum, all_sum,
                                                                                                  weights))
        return

    def feature_box(self, data_list):
        """
        wrapper for box plot for all features
        """
        column_names = data_list[0][1].columns
        for column_name in column_names:
            self.feature_box_single_column(column_name, data_list)
        return

    def feature_box_single_column(self, column_name, data_list):
        """
        box plot for single feature
        """
        class_names = {1: "W", 2: "1", 3: "2", 4: "3", 5: "R"}
        class_name_list = [0]
        filename = "plot/box_" + column_name + ".png"
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        features = []
        for data in data_list:
            label, data = data
            label = class_names[label]
            class_name_list.append(label)
            features.append(data[column_name])
        plt.boxplot(features)
        plt.title('Box Plot for feature {}'.format(column_name))
        checkered_boxes = np.arange(6)
        plt.xticks(checkered_boxes, class_name_list)
        plt.xlabel('Stages')
        plt.ylabel('Distribution')
        fig.savefig(filename, dpi=fig.dpi)
        return

    def split_data(self, data):
        """
        used in spark for box plot, flatmap
        """
        length = len(data[1])
        res = []
        for i in range(length):
            res.append((data[1][i], data[0].iloc[i, :]))
        return res

    def concat_series(self, series1, series2):
        """
        used in spark for box plot, reduce
        """
        index = series1.index
        res = pd.concat([series1, series2], axis=1, ignore_index=True)
        res.index = index
        return res

    def rnn_only_feature(self):
        """
        this function is used to produce box plot for training data of hand made cnn model
        spark is used so we can regroup data by feature name
        """
        findspark.init()
        spark_session = SparkSession.builder.appName("datapanalyzer").getOrCreate()
        sc = spark_session.sparkContext
        data = self.load_rnn_only_train()
        subject_rdd = sc.parallelize(data)
        subject_rdd_grouped_by_label = subject_rdd.flatMap(lambda batch: self.split_data(batch)).reduceByKey(lambda x1, x2: self.concat_series(x1, x2)).map(lambda x: (x[0], x[1].transpose()))
        list_of_features = subject_rdd_grouped_by_label.collect()
        self.feature_box(list_of_features)
        return

    def cnn_rnn_distribution(self):
        """
        get distribution among train, val and test set in cnn rnn model
        """
        data = self.load_cnn_rnn_train()
        bin_sum, all_sum, weights = self.single_distribution(data)
        print("For training data for cnn rnn, bin sum is {}, total sum is {}, weights are {}".format(bin_sum, all_sum,
                                                                                                     weights))
        data = self.load_cnn_rnn_val()
        bin_sum, all_sum, weights = self.single_distribution(data)
        print("For validation data for cnn rnn, bin sum is {}, total sum is {}, weights are {}".format(bin_sum, all_sum,
                                                                                                       weights))
        data = self.load_cnn_rnn_test()
        bin_sum, all_sum, weights = self.single_distribution(data)
        print("For test data for cnn rnn, bin sum is {}, total sum is {}, weights are {}".format(bin_sum, all_sum,
                                                                                                 weights))
        return


if __name__ == "__main__":
    dao = DataAccessor()
    dao.rnn_only_distribution()
    dao.cnn_rnn_distribution()
    # dao.get_rnn_only_weight()
    # dao.rnn_only_feature()

