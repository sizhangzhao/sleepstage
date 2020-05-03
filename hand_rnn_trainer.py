from base_trainer import *
from sleep_dataloader import *
from hand_rnn import *
from data_accessor import *
from rnn_only_data_post_processor import *
from padded_loss import *


class HandRNNTrainer(BaseTrainer):

    """
    This is the trainer for hand made RNN model
    comments please check base learner
    """

    def __init__(self, hidden_size, num_epoch, batch_size, log_every=100, lr=0.01, dropout_ratio=0.5,
                 clip_grad=0.5, evaluate_val_each_step=True, step=50):
        self.hidden_size = hidden_size
        self.train_data, self.val_data, self.test_data, self.weights = self.load_data()
        self.step = step
        super(HandRNNTrainer, self).__init__(num_epoch, batch_size, log_every, lr, dropout_ratio, clip_grad,
                                             evaluate_val_each_step)

    def load_data(self):
        dao = DataAccessor()
        train = dao.load_rnn_only_train()
        val = dao.load_rnn_only_val()
        test = dao.load_rnn_only_test()
        weights = dao.get_rnn_only_weight()
        return train, val, test, weights

    def create_model(self):
        dataloader = SleepDataLoader(self.train_data, self.val_data, self.test_data, RNNPostProcessor(self.device, max_length=self.step), self.weights)
        model = HandRNN(20, self.hidden_size, self.dropout_ratio, self.device, step=self.step)
        loss = PaddedCrossEntropyLoss()
        return dataloader, model, loss

    def update_tag(self):
        self.tag += "h" + str(self.hidden_size)