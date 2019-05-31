


class Cost():

    def __init__(self):
        self.history = {}

    def record(self, cost, epoch_number):
        self.history[epoch_number] = cost

    def costK(self, a, y, epoch_number):
        cost = (a - y) * (a - y)
        self.record(cost, epoch_number)
        return cost