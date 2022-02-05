from typing import Optional, Iterable

class HashMap:
    data: dict[int, list]

    def __init__(self, values: Optional[Iterable]=None):
        self.data = {}
        if values is not None:
            for item in values:
                self.add(item)

    def __contains__(self, item):
        if hash(item) in self.data:
            return item in self.data[hash(item)]
        return False

    def add(self, item):
        self.data.setdefault(hash(item), []).append(item)

    def __getitem__(self, item):
        return self.data.setdefault(hash(item), [])