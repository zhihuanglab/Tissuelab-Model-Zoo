from collections import OrderedDict, MutableMapping


class Cache(MutableMapping):
    ## Cache with limited maximum capacit. 
    ## This is a simplified LRU caching scheme, 
    #  when the cache is full and a new page is referenced which is not there in cache,
    # it will remove the least recently used frame to spare space for new page.
    ## source: https://stackoverflow.com/questions/2437617/how-to-limit-the-size-of-a-dictionary
    def __init__(self, maxlen, items=None):
        self._maxlen = maxlen
        self.d = OrderedDict()
        if items:
            for k, v in items:
                self[k] = v

    @property
    def maxlen(self):
        return self._maxlen

    def __getitem__(self, key):
        self.d.move_to_end(key)
        return self.d[key]

    def __setitem__(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        elif len(self.d) == self.maxlen:
            self.d.popitem(last=False)
        self.d[key] = value

    def __delitem__(self, key):
        del self.d[key]

    def __iter__(self):
        return self.d.__iter__()

    def __len__(self):
        return len(self.d)