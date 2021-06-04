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

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.best_actions = util.Counter()

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for state in states:
            self.values[state] = 0.0

        for i in range(iterations):
            self.values = self.iteration(states)

    def getValue(self, state):
        """
      Return the value of the state (computed in __init__).
    """
        return self.values[state]

    def iteration(self, states):
        new_values = util.Counter()
        for state in states:
            val = float("-inf")
            for action in self.mdp.getPossibleActions(state):
                qval = self.computeQValue(state, action)
                if qval > val:
                    val = qval
                    self.best_actions[state] = action
                new_values[state] = val
        return new_values

    def computeQValue(self, state, action):
        """
      Compute the Q-value of action in state from the
      value function stored in self.values.
    """
        "*** YOUR CODE HERE ***"
        q_val = 0
        for state_2, p in self.mdp.getTransitionStatesAndProbs(state, action):
            q_val += p * (self.discount * self.values[state_2] + self.mdp.getReward(state, action, state_2))
        return q_val

    def getPolicy(self, state):
        if self.mdp.isTerminal(state):
            return None
        else:
            return self.best_actions[state]

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)

    def getQValue(self, state, action):
        return self.computeQValue(state, action)

