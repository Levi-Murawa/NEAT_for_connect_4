import pickle
import random


def agent_moj(obs, config):
    board = obs['board']
    we = obs['mark']
    for i in range(len(board)):
        if board[i] == we:
            board[i] = 1
        elif board[i] != 0:
            board[i] = -1

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return random.choice(valid_moves)


def agent_moj_net1(obs, config):
    board = obs['board']
    we = obs['mark']
    for i in range(len(board)):
        if board[i] == we:
            board[i] = 1
        elif board[i] != 0:
            board[i] = -1

    with open("net1", "rb") as f:
        net1 = pickle.load(f)
    output1 = net1.activate(board)

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    while (True):
        max_value = max(output1)
        max_index = output1.index(max_value)
        if max_index in valid_moves:
            print(max_index)
            return max_index
        else:
            output1[max_index] = -1

def agent_moj_net2(obs, config):
    board = obs['board']
    we = obs['mark']
    for i in range(len(board)):
        if board[i] == we:
            board[i] = 1
        elif board[i] != 0:
            board[i] = -1

    with open("net2", "rb") as f:
        net2 = pickle.load(f)
    output2 = net2.activate(board)

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    while (True):
        max_value = max(output2)
        max_index = output2.index(max_value)
        if max_index in valid_moves:
            print(max_index)
            return max_index
        else:
            output2[max_index] = -1