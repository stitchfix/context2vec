import gensim
from gensim import utils

class ContextCorpus(object):
    def __init__(self, contexts, context_prefix='C', example_prefix='I'):
        """
        ContextCorpus(contexts)

        Parameters
        ---------
        contexts : dict of sets
            Each set in the dict should be an observation occurring within
            the same context. E.g., every set should contain all pins by
            one client or all fixes by one client.

        context_prefix : str, optional
            This string will be prepended to all context keys

        example_prefix : str, optional
            This string will be prepended to all non-context words

        Examples
        ------
        # If user 1 bought items 0, 1, 2, 3 and user 2
        # bought 3, 4, 5 and user bought item 10
        contexts = {1:{0,1,2,3}, 2:{3,4,5}, 3:{10}}
        cc = ContextCorpus(contexts)
        """
        self.contexts = contexts
        self.context_prefix = context_prefix
        self.example_prefix = example_prefix

    def __iter__(self):
        """Create 'sentences' that start with the context as a word
           and then also have items as words on the same line """
        for context, items in self.contexts.iteritems():
            line = [self.context_prefix + str(context)]
            line += [self.example_prefix + str(i) for i in items]
            yield line
