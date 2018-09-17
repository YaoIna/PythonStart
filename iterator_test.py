class IteratorTest:

    def __init__(self):
        self.list = [0, 1, 2, 3, 4, 5, 6]

    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        return self.list[item]
