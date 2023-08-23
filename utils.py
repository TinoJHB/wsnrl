
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def concatenate_experiences(experiences):
    states = torch.FloatTensor([exp.state for exp in experiences]).to(device)
    actions = torch.LongTensor([exp.action for exp in experiences]).unsqueeze(1).to(device)
    rewards = torch.FloatTensor([exp.reward for exp in experiences]).unsqueeze(1).to(device)
    next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(device)
    masks = torch.FloatTensor([exp.mask for exp in experiences]).unsqueeze(1).to(device)
    return (states, actions, rewards, next_states, masks)