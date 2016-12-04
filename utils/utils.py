import numpy as np

from params import AgentActionType, UserActionType


def get_index_of_max_element(arr):
    """Returns the index of the highest element in the 1D numpy.ndarray.
    Ties are broken randomly.

    Args:
        arr (1D numpy.ndarray): Array of elements.

    Returns:
        int: Index of the highest element.
    """
    secondary = np.random.random(arr.size)
    sort_indices = np.lexsort((secondary, arr))
    return sort_indices[-1]


def normalize_probabilities(probabilities):
    """Normalizes the probabilities so that they sum to one.

    If they don't -- due to precision issues -- the difference is added to
    some element until it does.

    Args:
        probabilities (numpy.ndarray): Vector of probabilities.
    """
    sum_of_probabilities = np.sum(probabilities)
    while sum_of_probabilities != 1.0:
        diff = 1.0 - sum_of_probabilities
        random_index = np.random.randint(len(probabilities))
        probabilities[random_index] += diff
        sum_of_probabilities = np.sum(probabilities)


def collect_statistics(user, agent, dialog_session, num_sessions):
    """Runs multiple dialog sessions between the user and the agent to collect
    statistics about user's actions.

    Args:
        user (:obj: User): The dialog user.
        agent (:obj: Agent): The dialog agent.
        dialog_session (DialogSession): The dialog session class
        num_sessions (int): Number of dialog sessions to execute.
    """
    user_actions = [user_action for user_action in UserActionType]
    user_action_map = {action: i for i, action in
                       enumerate(user_actions)}
    user_action_stats = {action_type: np.zeros(len(user_actions))
                         for action_type in AgentActionType}

    agent_actions = [agent_action for agent_action in AgentActionType]
    agent_action_map = {action: i for i, action in
                        enumerate(agent_actions)}
    agent_action_counts = np.zeros(len(agent_actions))

    # Run multiple dialog sessions to gather user's action statistics.
    for _ in xrange(num_sessions):
        # Reset the agent and the user.
        agent.reset()
        user.reset(reset_policy=False)  # Only reset state, not policy.
        # Create a new dialog session.
        session = dialog_session(user, agent)
        # Start the dialog session by having the dialog agent make the
        # first move.
        agent_action = session.ask_agent_to_start()
        user_action = None
        while not (agent_action is AgentActionType.CLOSE and
                   user_action is UserActionType.CLOSE):
            user_action, next_agent_action = session.execute_one_step()

            # Update action statistics
            user_action_index = user_action_map[user_action]
            user_action_stats[agent_action][user_action_index] += 1
            agent_action_index = agent_action_map[agent_action]
            agent_action_counts[agent_action_index] += 1

            agent_action = next_agent_action

    print user_action_stats
    print agent_action_counts
