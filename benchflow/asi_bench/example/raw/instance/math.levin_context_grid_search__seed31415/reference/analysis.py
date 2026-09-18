from lts_task_core import ReferenceContextPolicy


def make_policy(training_data, config=None):
    """Reference weighted log-space context policy for GT self-checks.

    Uses training trajectories, high-order local contexts, family weights,
    and multiplicative expert mixing through ReferenceContextPolicy.
    """
    return ReferenceContextPolicy(training_data)
