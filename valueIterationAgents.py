import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
      * Please read learningAgents.py before reading this.*
      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """

    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount=0.9, iterations=100):
        """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
          mdp.isTerminal(state)
    """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.best_actions = dict()

        states = self.mdp.getStates()

        for state in states:
            self.values[state] = 0.0

        for i in range(iterations):
            self.values = self.iteration(states)

    def iteration(self, states):
        """
        A single Value Iteration
        :param states: list of states
        :return: new state values
        """
        new_values = util.Counter()
        for state in states:
            max_val = float("-inf")
            possible_actions = self.mdp.getPossibleActions(state)
            if not possible_actions:  # terminal state doesn't need value update
                continue
            for action in possible_actions:
                action_value = 0
                for state_2, p in self.mdp.getTransitionStatesAndProbs(state, action):  # at state s doing action a
                    action_value += p * self.values[state_2]
                if action_value > max_val:  # new best action
                    max_val = action_value
                    self.best_actions[state] = action
            new_values[state] = self.mdp.getReward(state, None, None) + self.discount * max_val  # update new value
        return new_values

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getQValue(self, state, action):
        """
        The q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        q_val = 0
        for state_2, p in self.mdp.getTransitionStatesAndProbs(state, action):  # at state s doing action a
            q_val += p * self.values[state_2]
        q_val = self.mdp.getReward(state, action, None) + self.discount * q_val
        return q_val

    def getPolicy(self, state):
        """
        The policy is the best action in the given state
        according to the values computed by value iteration.
        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        else:
            return self.best_actions[state]

    def getAction(self, state):
        """Returns the policy at the state (no exploration)."""
        return self.getPolicy(state)
