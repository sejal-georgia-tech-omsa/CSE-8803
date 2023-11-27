import gensim


class LDA:
    def __init__(self):
        """
        Initialize LDA Class
        """
        pass

    def tokenize_words(self, inputs):
        '''
        Lowercase, tokenize and de-accent sentences to produce the tokens using simple_preprocess function from gensim.utils.

        Args:
            inputs: Input Data. List of N sentences.
        Returns:
            output: Tokenized list of sentences. List of N lists of tokens.
        '''
    	
        return [gensim.utils.simple_preprocess(x) for x in inputs]

    def remove_stopwords(self, inputs, stop_words):
        """
        Remove stopwords from tokenized words.

        Args:
          inputs: Tokenized list of sentences. List of N lists of tokens.
          stop_words: List of S stop_words. 

        Returns:
          output: Filtered tokenized list of sentences. List of N lists of tokens with stop words removed.
        """

        return [[token for token in x if token not in stop_words] for x in inputs]

    def create_dictionary(self, inputs):
        """
        Create dictionary and term document frequency for the input data using Dictionary class of gensim.corpora.

        Args:
            inputs: Filtered tokenized list of sentences. List of N lists of tokens with stop words removed.

        Returns:
            id2word: Gensim Dictionary of index to word map.
            corpus: Term document frequency for each word. List of N lists of tuples.
        """

        id2word = gensim.corpora.Dictionary(inputs)
        corpus = [id2word.doc2bow(text) for text in inputs]

        return id2word, corpus

    def build_LDAModel(self, id2word, corpus, num_topics=10):
        """
        Build LDA Model using LdaMulticore class of gensim.models.

        Args:
          id2word: Gensim Dictionary of index to word map.
          corpus: Term document frequency for each word. List of N lists of tuples.
          num_topics: Number of topics for modeling (int)

        Returns:
          lda_model: LdaMulticore instance.
        """

        lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
        return lda_model
