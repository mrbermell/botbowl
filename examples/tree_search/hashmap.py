from typing import Optional, Iterable, Protocol, List, Dict


class HasSimpleHash(Protocol):
    simple_hash: str


class HashMap:
    data: Dict[str, List[HasSimpleHash]]

    def __init__(self, values: Optional[Iterable[HasSimpleHash]]=None):
        self.data = {}
        if values is not None:
            for item in values:
                self.add(item)

    def __contains__(self, item: HasSimpleHash):
        if item.simple_hash in self.data:
            for possible_match in self.data[item.simple_hash]:
                if item is possible_match:
                    return True
        return False

    def add(self, item: HasSimpleHash):
        self.data.setdefault(item.simple_hash, []).append(item)

    def __getitem__(self, item):
        return self.data.setdefault(item.simple_hash, [])

    def __len__(self):
        return sum(map(len, self.data.values()))

    def __iter__(self):
        for item_list in self.data.values():
            for item in item_list:
                yield item
