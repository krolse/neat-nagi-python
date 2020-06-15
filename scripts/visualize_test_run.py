import pickle

import matplotlib.pyplot as plt
from easygui import fileopenbox
from matplotlib.lines import Line2D

from definitions import ROOT_PATH
from nagi.constants import YELLOW, GREEN, GOLD

alpha = 0.8


def get_handles(with_test: bool):
    handles = (Line2D([0], [0], marker='o', color='b', linewidth=0),
               Line2D([0], [0], color=YELLOW),
               Line2D([0], [0], color=GREEN))
    return (*handles, Line2D([0], [0], color=GOLD)) if with_test else handles


def get_labels(plot_type: str, with_test: bool):
    labels = (f'{plot_type}', f'average {plot_type}', f'max {plot_type}')
    return (*labels, f'test {plot_type}') if with_test else labels


if __name__ == '__main__':
    path = fileopenbox(default=f"{ROOT_PATH}/data/*test_run*.pkl")
    with open(path, 'rb') as file:
        data = pickle.load(file)

    with_test_data = True if 'test_result' in data[0] else False

    x = range(len(data))

    # Fitness plot
    fitnesses = [generation['fitnesses'] for generation in data.values()]
    average_fitnesses = [sum(generation.values()) / len(generation) for generation in fitnesses]
    max_fitnesses = [max(generation.values()) for generation in fitnesses]

    fig = plt.figure()
    plt.suptitle('Fitness')
    plt.ylim(0, 1)
    plt.ylabel('fitness')
    plt.xlabel('generation')
    for generation, fitness in enumerate([fitness.values() for fitness in fitnesses]):
        plt.scatter([generation for _ in range(len(fitness))], fitness, color='b', s=0.1)
    plt.plot(x, average_fitnesses, YELLOW)
    plt.plot(x, max_fitnesses, GREEN, alpha=alpha)
    if with_test_data:
        test_fitnesses = [generation['test_result'][1] for generation in data.values()]
        plt.plot(x, test_fitnesses, GOLD, alpha=alpha)
    plt.figlegend(get_handles(with_test_data), get_labels('fitness', with_test_data), loc='upper left')
    plt.plot()

    try:
        # Accuracy plot
        accuracies = [generation['accuracies'] for generation in data.values()]
        average_accuracies = [sum(generation.values()) / len(generation) for generation in accuracies]
        max_accuracies = [max(generation.values()) for generation in accuracies]

        fig = plt.figure()
        plt.suptitle('Accuracy')
        plt.ylim(0, 1)
        plt.ylabel('accuracy')
        plt.xlabel('generation')
        for generation, accuracy in enumerate([accuracy.values() for accuracy in accuracies]):
            plt.scatter([generation for _ in range(len(accuracy))], accuracy, color='b', s=0.1)
        plt.plot(x, average_accuracies, YELLOW)
        plt.plot(x, max_accuracies, GREEN, alpha=alpha)
        if with_test_data:
            test_accuracies = [generation['test_result'][2] for generation in data.values()]
            plt.plot(x, test_accuracies, GOLD, alpha=alpha)
        plt.figlegend(get_handles(with_test_data), get_labels('accuracy', with_test_data), loc='upper left')
        plt.plot()

        # End-of-sample accuracy plot
        end_of_sample_accuracies = [generation['end_of_sample_accuracies'] for generation in data.values()]
        average_end_of_sample_accuracies = [sum(generation.values()) / len(generation) for generation in
                                            end_of_sample_accuracies]
        max_end_of_sample_accuracies = [max(generation.values()) for generation in
                                        end_of_sample_accuracies]

        fig = plt.figure()
        plt.suptitle('End-of-sample Accuracy')
        plt.ylim(0, 1)
        plt.ylabel('end-of-sample accuracy')
        plt.xlabel('generation')
        for generation, accuracy in enumerate([accuracy.values() for accuracy in end_of_sample_accuracies]):
            plt.scatter([generation for _ in range(len(accuracy))], accuracy, color='b', s=0.1)
        plt.plot(x, average_end_of_sample_accuracies, YELLOW)
        plt.plot(x, max_end_of_sample_accuracies, GREEN, alpha=alpha)
        if with_test_data:
            test_end_of_sample_accuracies = [generation['test_result'][3] for generation in data.values()]
            plt.plot(x, test_end_of_sample_accuracies, GOLD, alpha=alpha)
        plt.figlegend(get_handles(with_test_data), get_labels('eos accuracy', with_test_data), loc='upper left')
        plt.plot()

    except KeyError:  # Older test run data don't contain accuracies
        pass

    plt.show()
