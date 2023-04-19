from kaggle_environments import make, evaluate
import neat
import os
import pickle
import random


def eval_genomes(genomes, config):
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            train_ai(genome1, genome2, config)


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

def agent_moj_net(obs, config):
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
    print(output1)
    
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return random.choice(valid_moves)
    

def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-7')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 2)
    #with open("best.pickle", "wb") as f:
    #    pickle.dump(winner, f)


def eval_std(agent = ["random","random"],num=50):
    reward = evaluate(agents = agent, environment = "connectx", num_episodes=num)
    ag1 = 0
    ag2 = 0
    for i in range(len(reward)):
        ag1 = ag1 + reward[i][0]
        ag2 = ag2 + reward[i][1]
    return(ag1, ag2)

def train_ai(genome1, genome2, config):
    net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
    net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
    with open("net1", "wb") as f:
        pickle.dump(net1, f)
    with open("net2", "wb") as f:
        pickle.dump(net2, f)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)
    
    env = make("connectx", debug=True)
    env.run([agent_moj_net, "random"])
    #gra = env.render(mode="ansi")
    #print(gra)

