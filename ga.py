import random
import numpy as np

"""
This file includes the genetic algorithm used to sub-select the best features to use in 
training and testing.
"""


def create_weighted_selection(pop):
    # Created a weighted population, where those feature sets with good (higher) fitness values are placed in
    # more times. Also, cull those feature sets in the population with terrible fitness values of <= .30.
    weighted_pop = []
    for feature_set in pop:
        if feature_set['fitness'] <= 0.3:
            pop.remove(feature_set)
        elif feature_set['fitness'] <= 0.35:
            weighted_pop += [feature_set]
        elif feature_set['fitness'] <= 0.4:
            weighted_pop += [feature_set] * 2
        elif feature_set['fitness'] <= 0.45:
            weighted_pop += [feature_set] * 3
        elif feature_set['fitness'] <= 0.50:
            weighted_pop += [feature_set] * 4
        elif feature_set['fitness'] <= 0.55:
            weighted_pop += [feature_set] * 5
        elif feature_set['fitness'] <= 0.60:
            weighted_pop += [feature_set] * 6
        elif feature_set['fitness'] > 0.65:
            weighted_pop += [feature_set] * 7
        else:
            print("Fitness not set yet.")
    return weighted_pop


def selection(weighted_pop):
    """
    Selects a feature_set from the weighted population randomly, biased toward those with better (higher)
    fitness values.
    :param pop: The population of feature sets to select from.
    :return: A single feature set from that population. Fitter feature sets have a higher chance of selection.
    """
    return random.choice(weighted_pop)


def reproduce(x, y, num_features):
    """
    Produces a child feature set from the two chromosomes x and y. The crossover point is chosen randomly based on the
    lengths of the chromosomes.
    :param x: The selected x chromosome, representing a feature set.
    :param y: The selected y chromosome, representing a feature set.
    :return: A child feature set, created from chromosomes x and y.
    """
    crossover_point = random.randint(0, num_features)
    spliced_x = np.array(x['set'][0: crossover_point])
    spliced_y = np.array(y['set'][crossover_point: num_features])

    child = {
        'id': None,
        'set': np.concatenate((spliced_x, spliced_y)),
        'feature_titles': None,
        'fitness': 0}

    return child


def mutate(feature_set, num_features):
    mutation_index = random.randint(0, num_features - 1)

    if feature_set['set'][mutation_index] == 1:
        feature_set['set'][mutation_index] = 0
    else:
        feature_set['set'][mutation_index] = 1

    return feature_set
