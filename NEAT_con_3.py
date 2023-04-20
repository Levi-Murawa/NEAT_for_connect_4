from kaggle_environments import make, evaluate
from Agents import agent_moj_net1, agent_moj_net2
import neat
import os
import pickle
import time
import random
start_time = time.time()
def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-7')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    #print("Wejscie do treningu:", time.time() - start_time)
    winner = p.run(eval_genomes, 1)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def eval_genomes(genomes, config):
    ile = 0
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i+1:]:
            ile = ile +1
            #print(ile, "///", genome_id1, genome_id2)
            #print("Kolejne genomy:", time.time() - start_time)
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            train_ai(genome1, genome2, config)


def train_ai(genome1, genome2, config):
    #print("Zapis:", time.time() - start_time)
    net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
    net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
    #print("Koniec zapisu:", time.time() - start_time)
    with open("net1", "wb") as f:
        pickle.dump(net1, f)
    with open("net2", "wb") as f:
        pickle.dump(net2, f)
    #print("Początek meczu:", time.time() - start_time)
    #env = make("connectx", debug=True)
    #env.run([agent_moj_net1, agent_moj_net2])
    (a, b) = eval_std([agent_moj_net1, agent_moj_net2], 3)
    #print("Koniec meczu:", time.time() - start_time)
    genome1.fitness += a
    genome2.fitness += b


def eval_std(agent = ["random","random"],num=50):
    reward = evaluate(agents = agent, environment = "connectx", num_episodes=num)
    ag1 = 0
    ag2 = 0
    for i in range(len(reward)):
        ag1 = ag1 + reward[i][0]
        ag2 = ag2 + reward[i][1]
    return(ag1, ag2)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)
    
    env = make("connectx", debug=True)
    env.run([agent_moj_net1, agent_moj_net2])
    gra = env.render(mode="ansi")
    print(gra)

