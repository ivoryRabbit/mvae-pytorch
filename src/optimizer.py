import torch


class Optimizer(object):
    def __init__(self, params, optimizer_type, lr: float):
        """
        Args:
            params: torch.nn.Parameter. The NN parameters to optimize
            optimizer_type: type of the optimizer to use
            lr: learning rate
        """
        if optimizer_type == "RMSProp":
            self.optimizer = torch.optim.RMSprop(params, lr=lr)
        elif optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(params, lr=lr)
        elif optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(params, lr=lr)
        else:
            raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self):
        return self.optimizer.load_state_dict()
