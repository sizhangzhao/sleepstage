from base_trainer import *
from sleep_dataloader import *
from sleep_cnn_rnn import *
from data_accessor import *
from cnn_rnn_data_post_processor import *
from padded_loss import *
from torch.nn import Parameter


class CNNRNNTrainer(BaseTrainer):

    """
    this is the trainer for CNN RNN model
    comments please check base learner
    """

    def __init__(self, small_kernel, large_kernel, channel_number, output_size, hidden_size, cnn, cnn_step, num_epoch, batch_size, log_every=100, lr=0.01, dropout_ratio=0.5,
                 clip_grad=0.5, evaluate_val_each_step=True, step=50):
        self.small_kernel = small_kernel
        self.large_kernel = large_kernel
        self.channel_number = channel_number
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.train_data, self.val_data, self.test_data, self.weights = self.load_data()
        self.step = step
        self.cnn_model = cnn
        self.cnn_step = cnn_step
        super(CNNRNNTrainer, self).__init__(num_epoch, batch_size, log_every, lr, dropout_ratio, clip_grad,
                                             evaluate_val_each_step)
        self.load_pretrained_model()

    def load_data(self):
        dao = DataAccessor()
        train = dao.load_cnn_rnn_train()
        val = dao.load_cnn_rnn_val()
        test = dao.load_cnn_rnn_test()
        weights = dao.get_cnn_rnn_weight()
        return train, val, test, weights

    def create_model(self):
        dataloader = SleepDataLoader(self.train_data, self.val_data, self.test_data, CNNRNNPostProcessor(self.device, max_length=self.step), self.weights)
        model = SleepCNNRNN(self.small_kernel, self.large_kernel, self.channel_number, self.output_size, self.hidden_size, self.dropout_ratio, self.device, step=self.step)
        loss = PaddedCrossEntropyLoss()
        return dataloader, model, loss

    def update_tag(self):
        self.tag += "sm" + str(self.small_kernel) + "la" + str(self.large_kernel) + "cn" + str(self.channel_number) + "op" + str(self.output_size) + "h" + str(self.hidden_size)

    def load_pretrained_model(self):
        """
        This is a special function for CNN-RNN model so we can load pretrained CNN model
        """
        print("Loading pretrained model")
        self.cnn_model.load_model(num_iter=self.cnn_step)
        state_dict = self.cnn_model.get_model_state_dict()
        self.model.load_state_dict(state_dict, strict=False)
        print("Finish loading pretrained model")
        return