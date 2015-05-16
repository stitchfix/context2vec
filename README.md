## Using Context2Vec

Use like:

    from sphere.models.context2vec import ContextCorpus
    from gensim.models import Word2Vec

    # If user 1 bought items 0, 1, 2, 3 and user 2
    # bought 3, 4, 5 and user bought item 10
    contexts = {1:{0, 1, 2, 3}, 2:{2, 3, 4, 5}, 3:{10}}
    cc = ContextCorpus(contexts)

    # Run the w2v model
    model = Word2Vec(cc, size=30, iter=100, min_count=1, alpha=0.025, sg=0)

    # Find the most similar user to user 1
    results = model.most_similar('C1', topn=100)
    related_users = filter(lambda x: x[0].startswith('C'), results)

    # Find all user vectors
    users = filter(lambda x: x.startswith('C'), model.index2word)
    user_vectors = [model.syn0[w2v.vocab[w].index] for w in users]

## Running tests
Simply run

    nosetests

Or to debug:

    nosetests --pdb
