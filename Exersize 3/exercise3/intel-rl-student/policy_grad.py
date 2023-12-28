import torch
from torch.autograd import Variable

'''
def compute_losses(R, ep_rewards, ep_values, ep_action_log_probs, masks, gamma, tau, algo):
    """ Compute the policy and value func losses given the terminal reward, the episode rewards,
    values, and action log probabilities """
    policy_loss = 0.0
    value_loss = 0.0

    # TODO: Use the available parameters to compute the policy gradient loss and the value function
    # loss.
    
    
    
    raise NotImplementedError("Compute the policy and value function loss.")
    return policy_loss, value_loss
'''


def compute_losses(R, ep_rewards, ep_values, ep_action_log_probs, masks, gamma, tau, algo):
    """ Compute the policy and value func losses given the terminal reward, the episode rewards,
    values, and action log probabilities """
    policy_loss = 0.0
    value_loss = 0.0

    # Calculate the advantages
    advantages = torch.tensor(R - ep_values, dtype=torch.float32).to(ep_values.device)
    
    # Compute the value loss
    value_targets = R
    value_loss = (ep_values - value_targets).pow(2).mean()
    
    # Compute the policy loss
    policy_loss = -torch.sum(advantages * ep_action_log_probs)
    
    # Optionally add entropy regularization to the policy loss
    if algo == 'PPO':
        entropy = -torch.sum(ep_action_log_probs * torch.exp(ep_action_log_probs))
        policy_loss -= 0.01 * entropy  # You can adjust the coefficient as needed
    
    return policy_loss, value_loss

