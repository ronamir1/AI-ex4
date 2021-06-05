# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):

    def __init__(self, **args):
        """You can initialize Q-values here..."""
        ReinforcementAgent.__init__(self, **args)
        self.q = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """
        if (state, action) not in self.q:
            return 0.0
        return self.q[(state, action)]

    def getValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legal_actions = self.getLegalActions(state)
        if not legal_actions:  # terminal state
            return 0.0
        return self.get_max_q(state)[1]

    def getPolicy(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return None
        return self.get_max_q(state)[0]

    def getAction(self, state):
        """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
        # Pick Action
        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return None
        if util.flipCoin(self.epsilon):  # random action
            return random.choice(legal_actions)
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
        "*** YOUR CODE HERE ***"
        self.q[(state, action)] += self.alpha * (
                reward + self.discount * self.get_max_q(nextState)[1] - self.q[(state, action)])

    def get_max_q(self, state):
        """
        Gets the max q value (action, value) pair of current state
        :param state: current state
        :return: (action, value) pair of max q value
        """
        max_q = float("-inf")
        max_action = []
        if not self.getLegalActions(state):  # terminal state
            return None, 0.0
        for action in self.getLegalActions(state):
            q_value = self.getQValue(state, action)
            if q_value == max_q:  # add for tie breaking
                max_action.append(action)
            if q_value > max_q:  # new max q value
                max_q = q_value
                max_action = [action]
        return random.choice(max_action), max_q


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.w = util.Counter()  # counter for weights; mapping features to weight values

        # You might want to initialize weights here.
        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
        "*** YOUR CODE HERE ***"
        return self.w * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        """
       Should update your weights based on transition
    """
        "*** YOUR CODE HERE ***"
        correction = (reward + self.discount * self.getValue(nextState) - self.getQValue(state, action))
        feature_vector = self.featExtractor.getFeatures(state, action)
        for feature in feature_vector:
            self.w[feature] += self.alpha * correction * feature_vector[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
