from blessings import Terminal
from progressbar import  ProgressBar
import progressbar
import sys
import time
from random import random
class TermLogger(object):
    def __init__(self, n_epochs, train_size, valid_size):
        self.n_epochs = n_epochs
        self.train_size = train_size
        self.valid_size = valid_size
        self.t = Terminal()
        space = 10#前面空10行
        be = 1  # epoch bar position
        bt = 3  # train bar position
        bv = 6  # valid bar position
        h = self.t.height
        if h ==None:
            h=24# 终端高度


        self.epoch_writer = Writer(self.t, (0, h-space))
        self.epoch_bar_wirter = Writer(self.t, (0, h-space+be))

        self.train_writer = Writer(self.t, (0, h-space+bt))#public
        self.train_bar_writer = Writer(self.t, (0, h-space+bt+1))

        self.valid_writer = Writer(self.t, (0, h-space+bv))#public
        self.valid_bar_writer = Writer(self.t, (0, h-space+bv+1))

        self.reset_epoch_bar()
        self.reset_train_bar()#152 batches
        self.reset_valid_bar()# 124 batches

    #private
    def reset_epoch_bar(self):
        self.epoch_bar = ProgressBar(maxval=self.n_epochs, fd=self.epoch_bar_wirter).start()
    #public
    def reset_train_bar(self):
        self.train_bar = ProgressBar(maxval=self.train_size, fd=self.train_bar_writer).start()
    #public
    def reset_valid_bar(self):
        self.valid_bar = ProgressBar(maxval=self.valid_size, fd=self.valid_bar_writer).start()
    #public
    def epoch_logger_update(self,epoch,str):
        self.epoch_bar.update(epoch)
        self.epoch_writer.write(str)
    def valid_logger_update(self,batch_i,str):
        self.valid_bar.update(batch_i + 1)  # key methods
        self.valid_writer.write(str)
    def train_logger_update(self,batch_i,str):
        self.train_bar.update(batch_i + 1)  # key methods
        self.train_writer.write(str)


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



def train(logger):
    batch_loss= random()
    for batch_i in range(logger.train_size):
        time.sleep(0.2)
        logger.train_logger_update(batch_i,
                                   'batch loss {} '.format(batch_loss))
    return 0
def val(logger):
    batch_loss= random()

    for batch_i in range(logger.valid_size):
        time.sleep(0.2)
        logger.valid_logger_update(batch_i,
                                   'batchloss {}'.format(batch_loss))

    return 0

def TermLogger_demo():
    '''
    TermLogger demo
    训练框架
    :return:
    '''
    epochs = 15
    train_size=10#batchs for train
    valid_size = 6


    logger = TermLogger(n_epochs=epochs,
                        train_size=train_size,
                        valid_size=valid_size)
    logger.reset_epoch_bar()


    #first val
    first_val = True
    if first_val:
        val_loss = val(logger)
    else:
        val_loss=0

    logger.reset_epoch_bar()
    logger.epoch_bar.update(epoch=0)
    logger.epoch_writer.write('epoch {} train loss{} val loss{}'.format(0, None, val_loss))

    for epoch in range(1,epochs):

        train_loss=train(logger)

        val_loss=val(logger)




        logger.reset_train_bar()
        logger.reset_valid_bar()
        logger.epoch_logger_update(epoch,'train loss {} val loss{}'.format(train_loss, val_loss))

    logger.epoch_bar.finish()
    print('over')

def progressbar_demo1():
    '''
    很像tqdm
    :return:
    '''
    total = 1000
    probar = ProgressBar()
    for i in probar(range(100)):
            time.sleep(0.01)
def progressbar_demo3():
    pass
    total = 100
    pbar =ProgressBar(maxval=total)
    for i in range(total):

        print(i)
        pbar.update(i)
        time.sleep(0.1)
    pbar.finish()

def tqdm_demo():
    '''
    很像tqdm
    :return:
    '''
    total = 1000
    from tqdm import tqdm
    for i in tqdm(range(100)):
        time.sleep(0.01)

def progressbar_demo4():
    import time
    import progressbar
    widgets = [ progressbar.Bar('#'),
                ' [', progressbar.Timer(), '] ',
                progressbar.Percentage(),
                '(', progressbar.ETA(), ') ',]

    bar = progressbar.ProgressBar(maxval=100,
                                  widgets=widgets)
    bar.start()
    for i in range(100):
        bar.update(i)

        time.sleep(0.01)


if __name__ =='__main__':
    TermLogger_demo()
    #progressbar_demo4()
    #tqdm_demo()