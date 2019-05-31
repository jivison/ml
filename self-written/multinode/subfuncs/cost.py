


class Cost():

    def __init__(self):
        self.history = {}

    def record(self, cost, epoch_number):
        self.history[epoch_number] = cost

    def costK(self, a, y):
        cost = (a - y) * (a - y)
        return cost

    