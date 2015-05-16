import unittest
from sphere.models.context2vec import ContextCorpus
from gensim.models import Word2Vec

class TestList2Vec(unittest.TestCase):
    def test_context_corpus_iter(self):
        """Check that the iter returns properly formatted lines"""
        contexts = {1:{0, 1, 2, 3}, 2:{3, 4, 5}, 3:{10}}
        cc = ContextCorpus(contexts)
        line = list(next(cc.__iter__()))
        context = int(line[0].replace('C', ''))
        examples = contexts[context]
        good = all((int(i.replace('I', '')) in examples for i in line[1:]))
        self.assertTrue(good)

    def test_context2vec_context_similarity(self):
        """Assert that the most similar user vector to user 1 is user 2"""
        contexts = {1:{0, 1, 2, 3}, 2:{2, 3, 4, 5}, 3:{10}}
        cc = ContextCorpus(contexts)
        model = Word2Vec(cc, size=3, iter=1000, min_count=1, alpha=0.1, sg=0)
        results = filter(lambda x: x[0].startswith('C'), model.most_similar('C1',
            topn=100))
        self.assertTrue(results[0][0] == 'C2')

    def test_context2vec_all_contexts(self):
        """Number of contexts in model should be same as input"""
        contexts = {1:{0, 1, 2, 3}, 2:{2, 3, 4, 5}, 3:{10}}
        cc = ContextCorpus(contexts)
        model = Word2Vec(cc, size=3, iter=1000, min_count=1, alpha=0.1, sg=0)
        users = filter(lambda x: x.startswith('C'), model.index2word)
        user_vectors = [model.syn0[model.vocab[w].index] for w in users]
        self.assertTrue(len(user_vectors) == len(contexts))
