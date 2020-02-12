############################################################
# CIS 521: Homework 1
############################################################

student_name = "Shubhankar Patankar"

############################################################
# Section 1: Python Concepts - Study Questions
############################################################

python_concepts_question_1 = """
Python is strongly typed because every object has a fixed type.
The interpreter prohibits operations on objects with incompatible types.
For instance, 'foo' + 2 would throw an error since a 'string' is being 
added to an 'integer'.
Python is dynamically typed because variables take on their type at the
moment of first assignment. For instance, x = 2 bestows the 'integer' type on 
the variable x, but a subsequent command of x = 'foo' changes this type to
'string'.
"""

python_concepts_question_2 = """
Dictionary keys must be immutable. Here we are trying to use lists as our keys.
To avoid this problem, we could cast each coordinate pair list as a tuple. This is
acceptable, since tuples are immutable.
"""

python_concepts_question_3 = """
The second approach uses an iterator to concatenate the strings. Iterators do
not load the entire input argument into memory, instead they only store
the state that denotes which strings have already been concatenated. The former
approach with the for loop requires that the entire input be loaded into memory
and then looped over, which may be significantly slower for large inputs.
"""

############################################################
# Section 2: Working with Lists
############################################################

def extract_and_apply(l, p, f):
    return [f(x) for x in l if p(x)]

def concatenate(seqs):
    return [elem for seq in seqs for elem in seq]

def transpose(matrix):
    num_lists_in_transpose = len(matrix[0])
    transpose_matrix = [] # initialize empty transpose matrix
    for i in range(num_lists_in_transpose): 
        list_for_transpose = [] # begin building each transpose list
        for original_list in matrix: 
            # index the element from the non-tranpose list and place it 
            # in the tranposed list corresponding to the index
            list_for_transpose.append(original_list[i])
        transpose_matrix.append(list_for_transpose)
    return transpose_matrix

############################################################
# Section 3: Sequence Slicing
############################################################

def copy(seq):
    return seq[:]

def all_but_last(seq):
    return seq[:-1]

def every_other(seq):
    return seq[:len(seq):2]

############################################################
# Section 4: Combinatorial Algorithms
############################################################

def prefixes(seq):
    current = 0
    while current <= len(seq):
        yield seq[0:current]
        current += 1
        
def suffixes(seq):
    start = 0
    end = len(seq)
    while start <= len(seq):
        yield seq[start:end]
        start += 1
        
def slices(seq):
    start = 0
    while start < len(seq): 
        end = start + 1
        while end <= len(seq):
            yield seq[start:end]
            end += 1
        start += 1

############################################################
# Section 5: Text Processing
############################################################

def normalize(text):
    split_text = text.split() # separate string into a list of units
    lowered_split = []
    for split in split_text:
        lowered_split.append(split.lower()) # `lower' every unit
    normalized = ' '.join(lowered_split) # join list of lowered units into string
    return normalized

def no_vowels(text):
    vowels = ['a','e','i','o','u','A','E','I','O','U']
    text = [elem for elem in text if elem not in vowels]
    return ''.join(text)

def digits_to_words(text):
    d = {'1':'one','2':'two','3':'three','4':'four',
        '5':'five','6':'six','7':'seven','8':'eight',
        '9':'nine','0':'zero'}
    return ' '.join([d[elem] for elem in text if elem.isdigit()])

def to_mixed_case(name):
    remove_unders = name.split('_') 
    # split by underscores, if more than one then empty element returned
    terms = [word.lower() for word in remove_unders if word] 
    # if `not empty' then lower the term
    if terms:
        term_0 = [terms[0]]
        # remove the first term, since it is not camel-cased
        terms = terms[1:len(terms)]
        capitalized = [term.capitalize() for term in terms]
        # capitalize all other terms
        mixed_terms = term_0 + capitalized # combine term_0 with others
        return ''.join(mixed_terms)
    else:
        return ''

############################################################
# Section 6: Polynomials
############################################################

class Polynomial(object):
    
    def __init__(self, polynomial):
        # initialize Polynomial
        self.polynomial = tuple(polynomial)
        
    def get_polynomial(self):
        return self.polynomial
    
    def __neg__(self):
        # for every tuple in polynomial, flip sign on the order 
        neg = [(-n,k) for (n,k) in self.get_polynomial()]
        return Polynomial(tuple(neg))
    
    def __add__(self, other):
        return Polynomial(self.polynomial + other.polynomial)
    
    def __sub__(self, other):
        neg_other = -other # first negate the Polynomial
        return Polynomial(self.polynomial + neg_other.polynomial)
    
    def __mul__(self, other):
        tuples_self = self.polynomial
        tuples_other = other.polynomial
        return_tuples = ()
        for pair in tuples_self:
            coeff = pair[0] # coefficient from tuple
            order = pair[1] # order from tuple
            for other_pair in tuples_other:
                other_coeff = other_pair[0] # coefficient from tuple
                other_order = other_pair[1] # order from tuple
                result_coeff = coeff * other_coeff # multiply the coeffs
                result_order = order + other_order # add the orders
                result_tuple = (result_coeff, result_order) # combine into new tuples
                return_tuples = return_tuples + (result_tuple,) # create new tuple of tuples
        return Polynomial(return_tuples) # create and return Polynomial object 
    
    def __call__(self, x):
        poly = self.polynomial
        result = 0 # initialize return value
        for pair in poly:
            coeff = pair[0]
            order = pair[1]
            result = result + (coeff * x**order)
        return result
    
    def simplify(self):
        sorted_tuples = sorted(self.get_polynomial(), key = lambda pair: pair[1], reverse = True)
        all_orders = [k for (n,k) in sorted_tuples]
        # get all order values from tuples
        all_coeffs = [n for (n,k) in sorted_tuples]
        # get all coefficient values from tuples
        return_tuples = () 
        for order in range(max(all_orders) + 1): # +1 to ensure last order is captured
            new_coeff = sum([elem for idx, elem in enumerate(all_coeffs) if all_orders[idx] == order])
            if new_coeff == 0:
                continue
            # sum those coefficients in all_coeffs which are in the positions corresponding to same order
            result_tuple = (new_coeff, order)
            return_tuples = return_tuples + (result_tuple,) # create new tuple of tuples
            # results are in ascending order of the polynomial order, need to be flipped
        if not return_tuples: # if all coefficients are 0, return tuple remains empty
            return_tuples = return_tuples + ((0,0),)
        return_tuples = return_tuples[::-1]
        self.polynomial = return_tuples

    def __str__(self):
        neg_sign = '-'
        pos_sign = '+'
        print_list = []
        for i, pair in enumerate(self.get_polynomial()):
            coeff = pair[0]
            order = pair[1]
            if coeff < 0: # negative coefficient 
                sign = neg_sign
                coeff = abs(coeff) # remove sign
            else: # non-negative coefficients have positive sign
                sign = pos_sign
            if i == 0: # first term:
                if sign == pos_sign:
                    # ignore '+' if first term is positive
                    if order == 0: # if order is zero, use only the coefficient
                        term_str = '%s' % (str(coeff))
                        print_list.append(term_str)
                    elif order == 1: # if order is one, omit power string
                        if coeff == 1:
                            term_str = 'x'
                            print_list.append(term_str)
                        else:
                            term_str = '%sx' % (str(coeff))
                            print_list.append(term_str)
                    else:
                        if coeff == 1:
                            term_str = 'x^%s' % (str(order))
                            print_list.append(term_str)
                        else:
                            term_str = '%sx^%s' % (str(coeff), str(order))
                            print_list.append(term_str)
                else:
                    if order == 0:
                        term_str = '%s%s' % (sign, str(coeff))
                        print_list.append(term_str)
                    elif order == 1: # if order is one, omit power string
                        if coeff == 1:
                            term_str = '%sx' % sign
                            print_list.append(term_str)
                        else:
                            term_str = '%s%sx' % (sign, str(coeff))
                            print_list.append(term_str)
                    else:
                        if coeff == 1:
                            term_str = '%sx^%s' % (sign, str(order))
                            print_list.append(term_str)
                        else:
                            term_str = '%s%sx^%s' % (sign, str(coeff), str(order))
                            print_list.append(term_str)
                continue # to avoid re-appending from following line
            if order == 0:
                term_str = '%s %s' % (sign, str(coeff))
                print_list.append(term_str)
            elif order == 1:
                if coeff == 1:
                    term_str = '%s x' % sign
                    print_list.append(term_str)
                else:
                    term_str = '%s %sx' % (sign, str(coeff))
                    print_list.append(term_str)
            else:
                if coeff == 1:
                    term_str = '%s x^%s' % (sign, str(order))
                    print_list.append(term_str)
                else:
                    term_str = '%s %sx^%s' % (sign, str(coeff), str(order))
                    print_list.append(term_str)
        return ' '.join(print_list)
