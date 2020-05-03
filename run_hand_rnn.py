"""
Usage:
    run.py test --step=<int>
    run.py train
    run.py val --step=<int>

Options:
    -h --help                               show this screen.
    --step=<int>                            which step of parameter to be loaded
"""
from docopt import docopt
from hand_rnn_trainer import *


class Config:
    num_epoch = 1
    hidden_size = 18
    clip_grad = 0.5
    lr = 0.15
    dropout_ratio = 0.6
    log_every = 1
    batch_size = 32

# tensorboard --logdir C:\sleep\log --host=127.0.0.1
# python -m torch.utils.bottleneck run.py train
# python "C:\Users\szzha\Anaconda3\Scripts\kernprof-script.py" -l -v run.py train


if __name__ == '__main__':

    args = docopt(__doc__)

    runner = HandRNNTrainer(Config.hidden_size, num_epoch=1000, batch_size=64, log_every=100, lr=0.0001, dropout_ratio=0.5)

    if args['test']:
        runner.load_model(num_iter=args["--step"])
        runner.plot_train()
        runner.plot_val()
        runner.test()
    elif args['val']: # debug purpose, should never use purely
        runner.load_model(num_iter=args["--step"])
        runner.set_val_each_step(True)
        runner.validate()
    elif args['train']:
        # runner.load_model(num_iter=54900)
        runner.set_val_each_step(True)
        runner.train()
        # runner.test()
        # runner.set_val_each_step(False)
        # runner.validate()
