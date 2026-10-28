import torch

STATE_DIM = 64


def make_mixed(n, dim=STATE_DIM):
    a = torch.randint(0, 10, (n,))
    b = torch.randint(0, 10, (n,))
    c = torch.randint(0, 2, (n,))  # 0 или 1

    state = torch.zeros(n, dim)
    state[:, 0] = a.float() / 9.0
    state[:, 1] = b.float() / 9.0
    state[:, 2] = c.float()

    # c=0: a+b, c=1: a*b
    target = torch.where(c == 0, (a + b).float(), (a * b).float()) / 81.0

    return state, target.unsqueeze(-1)


def make_sqrt(n, dim=STATE_DIM):
    a = torch.rand(n) * 9.0 + 1.0  # [1, 10]

    state = torch.zeros(n, dim)
    state[:, 0] = a / 10.0
    state[:, 1] = 1.0  # начальное приближение x0 = 1.0
    state[:, 2] = 0.0  # счётчик шагов

    target = torch.sqrt(a).unsqueeze(-1) / 3.2

    return state, target


def make_conditional(n, dim=STATE_DIM):
    a = torch.randint(0, 10, (n,)).float()
    b = torch.randint(0, 10, (n,)).float()

    state = torch.zeros(n, dim)
    state[:, 0] = a / 9.0
    state[:, 1] = b / 9.0

    target = torch.where(a > b, a * b, a + b).unsqueeze(-1) / 81.0

    return state, target


def make_conditional2(n, dim=STATE_DIM):
    a = torch.randint(0, 10, (n,)).float()
    b = torch.randint(0, 10, (n,)).float()
    c = torch.randint(0, 10, (n,)).float()
    d = torch.randint(0, 10, (n,)).float()

    state = torch.zeros(n, dim)
    state[:, 0] = a / 9.0
    state[:, 1] = b / 9.0
    state[:, 2] = c / 9.0
    state[:, 3] = d / 9.0

    part1 = torch.where(a > b, a * b, a + b)
    part2 = torch.where(c > d, c * d, c + d)
    target = (part1 + part2).unsqueeze(-1) / 162.0

    return state, target


def make_sequential_sum(n, dim=STATE_DIM, n_numbers=8):
    numbers = torch.randint(0, 10, (n, n_numbers)).float()

    state = torch.zeros(n, dim)
    state[:, 0] = 0.0                          # acc
    state[:, 1] = 0.0                          # pos
    state[:, 2:2 + n_numbers] = numbers / 9.0  # x1..x8

    target = numbers.sum(dim=1, keepdim=True) / (9.0 * n_numbers)

    return state, target


def make_sort(n, dim=STATE_DIM, n_numbers=8):
    numbers = torch.randint(0, 10, (n, n_numbers)).float()

    state = torch.zeros(n, dim)
    state[:, :n_numbers] = numbers / 9.0

    # минимум — частичная проверка сортировки
    target = torch.sort(numbers, dim=1).values[:, 0:1] / 9.0

    return state, target