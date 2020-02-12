############################################################
# CIS 521: Homework 8
############################################################

student_name = "Shubhankar Patankar"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import re, string
from collections import defaultdict
import random
import math

############################################################
# Section 1: Markov Models
############################################################

def tokenize(text):
    tokens = re.findall(r"[\w]+|[^\s\w]", text)
    return tokens

def ngrams(n, tokens):
    prepend_token = '<START>'
    append_token = '<END>'
    for i in range(n - 1):
        tokens = [prepend_token] + tokens
    tokens.append(append_token)
    grams = []
    for j in range(len(tokens)-n+1):
        context = tuple(tokens[j:j+n-1])
        token = tokens[j+n-1]
        grams.append((context, token))
    return grams

class NgramModel(object):

    def __init__(self, n):
        self.n = n
        self.context_counts = defaultdict(lambda:0) # frequency of context
        self.sequence_counts = defaultdict(lambda:0) # frequency of (context, token) sequence

    def update(self, sentence):
        all_ngrams = ngrams(self.n, tokenize(sentence))
        for (context, token) in all_ngrams:
            self.context_counts[context] += 1 # increment the context count
            self.sequence_counts[(context, token)] += 1 # increment the (context, token) sequence count

    def prob(self, context, token):
        denominator = self.context_counts[context] # frequency of context followed by any token
        numerator = self.sequence_counts[(context,token)] # frequency of exact (context, token) sequence
        prob = numerator/denominator
        return prob

    def random_token(self, context):
        r = random.random()
        tokens_in_context = [] # collect all tokens that share the context
        for k, v in self.sequence_counts.items():
            # k is a (context, token) tuple
            # v is the count
            if k[0] == context: # found an n-gram with same context
                tokens_in_context.append(k[1])
        tokens_in_context = sorted(tokens_in_context) # enforce natural Pythonic ordering by sorting
        pre_sum = 0
        for i, token in enumerate(tokens_in_context):
            # pre_sum is sum of probabilities up to, but excluding, the current token
            post_sum = pre_sum + self.prob(context, token)
            # post_sum also includes the probability of the current token
            if pre_sum <= r < post_sum:
                return token
            pre_sum = post_sum
    
    def random_text(self, token_count):
        output_text = []
        all_context = ['<START>' for i in range(self.n - 1)] # keep a running context list initialized with <START>s
        for i in range(token_count):
            curr_context = tuple(all_context[len(all_context)-self.n+1:]) # extract context from running context list
            next_token = self.random_token(curr_context)
            output_text.append(next_token)
            if next_token == "<END>":
                # if next token is <END> append multiple <START>s to the running context list
                for i in range(self.n - 1):
                    all_context.append('<START>')
            else:
                # otherwise append the latest token to the running context list
                all_context.append(next_token)
        return " ".join(output_text)

    def perplexity(self, sentence):
        tokens = tokenize(sentence)
        m = len(tokens)
        all_ngrams = ngrams(self.n, tokens)
        log_prob_sum = 0
        for (context, token) in all_ngrams:
            log_prob_sum += math.log(self.prob(context, token))
        perplexity = math.exp(-1/(m+1) * log_prob_sum)
        return perplexity

def create_ngram_model(n, path):
    m = NgramModel(n)
    with open(path) as f:
        for line in f:
            m.update(line)
    return m

############################################################
# Section 2: Feedback
############################################################

feedback_question_1 = 10

feedback_question_2 = """
The random token generation was not very well explained, and I would have liked to understand its mechanics a little bit more.
"""

feedback_question_3 = """
I liked that at the end of the assignment, I had the tools to completely construct a language model.
I would have liked to have recieved some guidance on how to construct corpora by scraping the web, or information on other corpora that I could work with. 
"""
