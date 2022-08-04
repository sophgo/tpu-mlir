import abc

class base_class(metaclass=abc.ABCMeta):
    def __init__(self,args):
        self.init(args)

    @abc.abstractmethod
    def preproc(self, img_paths):
        pass

    @abc.abstractmethod
    def update(self, idx, outputs, img_paths = None, labels = None, ratios = None):
        pass

    @abc.abstractmethod
    def get_result(self):
        pass

    @abc.abstractmethod
    def print_info(self):
        pass
