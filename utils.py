import torch


def batch_accuracy(logits, labels):
    """
    follow Bilinear Attention Networks https://github.com/jnhwkim/ban-vqa.git
    """
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)

    return scores.sum(1)


def print_dict(d):
    for k in d:
        print("{}:{}".format(k,d[k]))


def print_lr(optimizer, prefix, epoch):
    all_rl = []
    for p in optimizer.param_groups:
        all_rl.append(p['lr'])
    print('{} E{:03d}:'.format(prefix, epoch), ' Learning Rate: ', set(all_rl))

def set_lr(optimizer, value):
    for p in optimizer.param_groups:
        p['lr'] = value

def decay_lr(optimizer, rate):
    for p in optimizer.param_groups:
        p['lr'] *= rate


class Tracker:
    """ Keep track of results over time, while having access to monitors to display information about them. """
    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        """ Track a set of results with given monitors under some name (e.g. 'val_acc').
            When appending to the returned list storage, use the monitors to retrieve useful information.
        """
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        # turn list storages into regular lists
        return {k: list(map(list, v)) for k, v in self.data.items()}


    class ListStorage:
        """ Storage of data points that updates the given monitors """
        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        """ Take the mean over the given values """
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        """ Take an exponentially moving mean over the given values """
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value