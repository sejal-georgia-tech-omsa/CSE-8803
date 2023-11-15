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
    	
        raise NotImplementedError

    def remove_stopwords(self, inputs, stop_words):
        """
        Remove stopwords from tokenized words.

        Args:
          inputs: Tokenized list of sentences. List of N lists of tokens.
          stop_words: List of S stop_words. 

        Returns:
          output: Filtered tokenized list of sentences. List of N lists of tokens with stop words removed.
        """

        raise NotImplementedError

    def create_dictionary(self, inputs):
        """
        Create dictionary and term document frequency for the input data using Dictionary class of gensim.corpora.

        Args:
            inputs: Filtered tokenized list of sentences. List of N lists of tokens with stop words removed.

        Returns:
            id2word: Gensim Dictionary of index to word map.
            corpus: Term document frequency for each word. List of N lists of tuples.
        """

        raise NotImplementedError

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

        raise NotImplementedError
