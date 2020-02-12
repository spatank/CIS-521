# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question6():
    # answerEpsilon = None
    # answerLearningRate = None
    # return answerEpsilon, answerLearningRate
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'

############
# Feedback #
############

# Just an approximation is fine.
feedback_question_1 = 10

feedback_question_2 = """
It was tricky to think through the implications of the changes in the learning rate and exploration parameters for the Bridge Crossing Revisited problem. 
It took me a few minutes to realize that the competing early- and late-stage demands of the search problem meant that the answer was 'NOT POSSIBLE'.
"""

feedback_question_3 = """
While it is definitely fun to watch the PacmanAgent, that part of the problem could be done without any added effort. 
I completed the first part of the assignment, and ended up getting full credit for the Pacman problem at the same time.
I watched the Pacman agent training only out of curiousity and could have completely ignored the whole section, which I think is a little weird.
"""

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
