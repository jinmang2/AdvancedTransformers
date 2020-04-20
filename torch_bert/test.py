class Test:
    # kwargs pop test - 200420
    def __init__(self, **kwargs):
        print(kwargs)
        self.pos = kwargs.pop('pos', 5)
        self.image = kwargs.pop('image', 'i love you')

    # class_method argument feeding test - 200420
    @classmethod
    def from_pretrained(cls, *input, **kwargs):
        return cls._from_pretrained(*input, **kwargs)

    @classmethod
    def _from_pretrained(cls, pretrained_model_name, cache_dir=None, *input, **kwargs):
        print(cls.prep)
        print(pretrained_model_name)
        print(cache_dir)
        return None

class BertTest(Test):

    prep = ['lol lol lol']

    def __init__(self, **kwargs):
        super().__init__()


if __name__ == '__main__':
    B = BertTest()
    B.from_pretrained('a')
