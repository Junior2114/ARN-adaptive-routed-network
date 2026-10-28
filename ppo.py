import torch


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    steps, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(B, device=rewards.device)
    for t in reversed(range(steps)):
        next_value = 0.0 if t == steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def normalize_advantages(advantages, clip=3.0):
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    adv = torch.clamp(adv, -clip, clip)
    return adv.detach()


def ppo_loss(log_probs, old_log_probs, advantages, entropies,
             clip_eps=0.2, entropy_coef=0.03):
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    entropy_loss = -entropies.mean()
    return policy_loss + entropy_coef * entropy_loss


def compute_approx_kl(log_probs, old_log_probs):
    return ((old_log_probs - log_probs).exp() - 1 - (old_log_probs - log_probs)).mean()