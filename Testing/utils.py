import seaborn as sns
from matplotlib import pyplot as plt

def get_iter(func):
    record = []
    for i in range(1,11):
        print('iteration',i, ', num of episodes ', i*100)
        record.append(func(episodes = i*100))

    x = [x*100 for x in range(1,len(record)+1)]
    sns.lineplot(x,record)
    plt.title('DQN with exploration')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.show()