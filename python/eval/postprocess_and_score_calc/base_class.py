import abc

class base_class(metaclass=abc.ABCMeta):
    def __init__(self,args):
        self.init(args)

    @abc.abstractmethod
    def update(self, idx, outputs, labels = None):
        pass

    @abc.abstractmethod
    def get_result(self):
        pass

    @abc.abstractmethod
    def print_info(self):
        pass
