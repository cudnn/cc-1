from blessings import Terminal
import progressbar
import sys
import time

class TermLogger(object):
    def __init__(self, n_epochs, train_size, valid_size):
        self.n_epochs = n_epochs
        self.train_size = train_size
        self.valid_size = valid_size
        self.t = Terminal()
        s = 10
        e = 1   # epoch bar position
        tr = 3  # train bar position
        ts = 6  # valid bar position
        #h = self.t.height
        h=24# 终端高度

        for i in range(10):
            print('')
        self.epoch_bar = progressbar.ProgressBar(maxval=n_epochs, fd=Writer(self.t, (0, h-s+e)))

        self.train_writer = Writer(self.t, (0, h-s+tr))
        self.train_bar_writer = Writer(self.t, (0, h-s+tr+1))

        self.valid_writer = Writer(self.t, (0, h-s+ts))
        self.valid_bar_writer = Writer(self.t, (0, h-s+ts+1))

        self.reset_train_bar()#152 batches
        self.reset_valid_bar()# 124 batches

    def reset_train_bar(self):
        self.train_bar = progressbar.ProgressBar(maxval=self.train_size, fd=self.train_bar_writer).start()

    def reset_valid_bar(self):
        self.valid_bar = progressbar.ProgressBar(maxval=self.valid_size, fd=self.valid_bar_writer).start()


class Writer(object):
    """Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    """

    def __init__(self, t, location):
        """
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        """
        self.location = location
        self.t = t

    def write(self, string):
        with self.t.location(*self.location):
            sys.stdout.write("\033[K")
            print(string)

    def flush(self):
        return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0]*i
        self.avg = [0]*i
        self.sum = [0]*i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert(len(val) == self.meters)
        self.count += n
        for i,v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)



def main():
    '''
    TermLogger test

    :return:
    '''
    epochs = 3
    train_size=10
    valid_size = 10
    logger = TermLogger(n_epochs=epochs,
                        train_size=train_size,
                        valid_size=valid_size)
    #logger.epoch_bar.start()

    for epoch in range(epochs):
        logger.epoch_bar.update(epoch)
        #logger.reset_train_bar()

        for batch_i in range(train_size):
            time.sleep(0.2)
            logger.train_bar.update(batch_i + 1)#key methods
            logger.train_bar_writer.write('train: Time {} Data {} Loss {}'.format(batch_i, 1.0, 0.2))

        for batch_i in range(train_size):
            time.sleep(0.2)
            logger.valid_bar.update(batch_i + 1)
            logger.valid_bar_writer.write('Valid: Time {} Data {} Loss {}'.format(batch_i, 1.0, 0.2))

        logger.epoch_bar.finish()

if __name__ =='__main__':
    main()