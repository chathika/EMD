"""EvolutionaryModelDiscovery: Automated agent rule generation and 
importance evaluation for agent-based models with Genetic Programming.
Copyright (C) 2018  Chathika Gunaratne
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>."""

from typing import Callable, Any, List, Dict
import multiprocessing
import random
from inspect import isclass

from deap import algorithms, gp, creator, base, tools
import pandas as pd
import numpy as np

from .Util import *
from .ABMEvaluator import (
    set_objective_function,
    set_model_factors,
    set_model_init_data,
    set_netlogo_writer,
    evaluate,
)
from .NetLogoWriter import NetLogoWriter


class SimpleDEAPGP:
    def __init__(
        self,
        model_init_data: Dict,
        ModelFactors: "EvolutionaryModelDiscovery.ModelFactors",
        netlogo_writer: NetLogoWriter,
    ) -> None:
        """
        Genetic program that handles the evolution of the agent-based model rule.

        :param model_init_data: Dict of model initialization properties.
        :param ModelFactors: ModelFactors module generated by EvolutionaryModelDiscovery 
                                by parsing the annotated NetLogo model files
        """

        self._mutation_rate = 0.2
        self._crossover_rate = 0.8
        self._generations = 10
        self._run_count = 1
        self._pop_init_size = 5
        set_model_init_data(model_init_data)
        set_model_factors(ModelFactors)
        set_netlogo_writer(netlogo_writer)
        self._pset = ModelFactors.get_DEAP_primitive_set()
        self._setup_DEAP_GP()

    def _setup_DEAP_GP(self) -> None:
        """
        Sets up the DEAP GP.

        """
        # Setting up genetic program
        # Setup DEAP GP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create(
            "IndividualMin", gp.PrimitiveTree, fitness=creator.FitnessMin
        )
        creator.create(
            "IndividualMax", gp.PrimitiveTree, fitness=creator.FitnessMax
        )
        self._toolbox = base.Toolbox()
        # Attribute generator
        self._toolbox.register(
            "expr_init", genGrow, pset=self._pset, min_=2, max_=10
        )
        # Structure initializers
        self._toolbox.register(
            "individual",
            tools.initIterate,
            creator.IndividualMin,
            self._toolbox.expr_init,
        )
        self._toolbox.register(
            "population", tools.initRepeat, list, self._toolbox.individual
        )
        # self._toolbox.register("setupCommands", tools.initRepeat, setupCommands )
        # self._toolbox.register("evaluate", self.evaluate)
        self._toolbox.register("select", tools.selTournament, tournsize=3)
        self._toolbox.register("mate", gp.cxOnePoint)
        self._toolbox.register("expr_mut", genGrow, min_=2, max_=3)
        self._toolbox.register(
            "mutate",
            gp.mutUniform,
            expr=self._toolbox.expr_mut,
            pset=self._pset,
        )
        self.hof = tools.HallOfFame(1)
        # global self._stats
        self._stats = tools.Statistics(get_values)  #
        self._stats.register("avg", np.mean)
        self._stats.register("std", np.std)
        self._stats.register("min", np.min)
        self._stats.register("max", np.max)
        # Genetic program setup successfully

    def set_mutation_rate(self, mutation_rate: float) -> None:
        self._mutation_rate = mutation_rate

    def set_crossover_rate(self, crossover_rate: float) -> None:
        self._crossover_rate = crossover_rate

    def set_generations(self, generations: int) -> None:
        self._generations = generations

    def set_population_size(self, population_size: int) -> None:
        self._pop_init_size = population_size

    def set_objective_function(self, objective_function: Callable) -> None:
        set_objective_function(objective_function)

    def set_depth(self, min: int, max: int) -> None:
        self._toolbox.register(
            "expr_init", genGrow, pset=self._pset, min_=min, max_=max
        )
        self._toolbox.register(
            "expr_mut", genGrow, pset=self._pset, min_=min, max_=max
        )

    def set_is_minimize(self, is_minimize: bool):
        if is_minimize:
            self._toolbox.register(
                "individual",
                tools.initIterate,
                creator.IndividualMin,
                self._toolbox.expr_init,
            )
        else:
            self._toolbox.register(
                "individual",
                tools.initIterate,
                creator.IndividualMax,
                self._toolbox.expr_init,
            )
        self._toolbox.register(
            "population", tools.initRepeat, list, self._toolbox.individual
        )

    def evolve(
        self, num_procs: int = multiprocessing.cpu_count(), verbose=__debug__
    ):
        """
        Chathika: made logging, stat collection, and multiprocessing related
        changes to this functions
        """
        """This algorithm reproduce the simplest evolutionary algorithm as
        presented in chapter 7 of [Back2000]_.

        :param verbose: Whether or not to log the statistics.
        :param num_procs: number of processes.
        :returns: The final population
        :returns: A class:`~deap.tools.Logbook` with the statistics of the
                evolution
        :returns: The factor scores in a pandas dataframe. 

        The algorithm takes in a population and evolves it in place using the
        :meth:`varAnd` method. It returns the optimized population and a
        :class:`~deap.tools.Logbook` with the statistics of the evolution. The
        logbook will contain the generation number, the number of evalutions for
        each generation and the statistics if a :class:`~deap.tools.Statistics` is
        given as argument. The *crossover_rate* and *mutation_rate* arguments are passed to the
        :func:`varAnd` function. The pseudocode goes as follow ::

            evaluate(population)
            for g in range(self._generations):
                population = select(population, len(population))
                offspring = varAnd(population, toolbox, crossover_rate, mutation_rate)
                evaluate(offspring)
                population = offspring

        As stated in the pseudocode above, the algorithm goes as follow. First, it
        evaluates the individuals with an invalid fitness. Second, it enters the
        generational loop where the selection procedure is applied to entirely
        replace the parental population. The 1:1 replacement ratio of this
        algorithm **requires** the selection procedure to be stochastic and to
        select multiple times the same individual, for example,
        :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
        Third, it applies the :func:`varAnd` function to produce the next
        generation population. Fourth, it evaluates the new individuals and
        compute the statistics on this population. Finally, when *self._generations*
        generations are done, the algorithm returns a tuple with the final
        population and a :class:`~deap.tools.Logbook` of the evolution.

        .. note::

            Using a non-stochastic selection method will result in no selection as
            the operator selects *n* individuals from a pool of *n*.

        This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
        :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
        registered in the toolbox.

        .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
        Basic Algorithms and Operators", 2000.
        """
        population = self._toolbox.population(n=self._pop_init_size)
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (
            self._stats.fields if self._stats else []
        )
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        factorScores = pd.DataFrame()
        num_procs = multiprocessing.cpu_count() if num_procs < 1 else num_procs

        with multiprocessing.pool.ThreadPool(num_procs) as pool:
            results = list(pool.imap(evaluate, invalid_ind))
        fitnesses = []
        factorScores = pd.DataFrame()
        for result in results:
            fitnesses.append(result.Fitness)
            fs = result
            fs["Gen"] = 0
            fs["Fitness"] = fs.Fitness[0]
            factorScores = factorScores.append(
                fs, ignore_index=True, sort=False
            )
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if self.hof is not None:
            self.hof.update(population)

        record = self._stats.compile(population) if self._stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, self._generations + 1):
            # Select the next generation individuals
            offspring = self._toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = algorithms.varAnd(
                offspring,
                self._toolbox,
                self._crossover_rate,
                self._mutation_rate,
            )
            for off in offspring:
                del off.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            with multiprocessing.pool.ThreadPool(num_procs) as pool:
                results = list(pool.imap(evaluate, invalid_ind))
            fitnesses = []
            for result in results:
                fitnesses.append(result.Fitness)
                fs = result
                fs["Gen"] = gen
                fs["Fitness"] = fs.Fitness[0]
                factorScores = factorScores.append(
                    fs, ignore_index=True, sort=False
                )
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if self.hof is not None:
                self.hof.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = self._stats.compile(population) if self._stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)
            # purge(".",".*.EMD.nlogo")
        return (
            population,
            logbook,
            factorScores,
        )


def genGrow(pset, min_: int, max_: int, type_: Any = None) -> List[Any]:
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a a node should be a terminal.
        """
        return depth == height or (
            depth >= min_ and random.random() < pset.terminalRatio
        )

    return generate(pset, min_, max_, condition, type_)


def generate(
    pset, min_: int, max_: int, condition: Callable, type_: Any = None
) -> List[Any]:
    """Generate a Tree as a list of list. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                    the height of the tree to build and the current
                    depth in the tree.
    :param type_: The type that should return the tree when called, when
                :obj:`None` (default) the type of :pset: (pset.ret)
                is assumed.
    :returns: A grown tree with leaves at possibly different depths
            dependending on the condition function.
    :raises: TypeError if no primitives or terminals of required type
    """
    validTree = False
    while not validTree:
        if type_ is None:
            type_ = pset.ret
        expr = []
        height = random.randint(min_, max_)
        stack = [(0, type_)]
        validTree = True
        while len(stack) != 0:
            depth, type_ = stack.pop()
            if condition(height, depth):
                if len(pset.terminals[type_]) > 0:
                    term = random.choice(pset.terminals[type_])
                    if isclass(term):
                        term = term()
                    expr.append(term)
                else:
                    if len(pset.primitives[type_]) > 0:
                        prim = random.choice(pset.primitives[type_])
                        expr.append(prim)
                        for arg in reversed(prim.args):
                            stack.append((depth + 1, arg))
                    else:
                        raise TypeError(
                            f"Invalid tree! No primitives or terminals of type {type_}"
                        )
            else:
                if len(pset.primitives[type_]) > 0:
                    prim = random.choice(pset.primitives[type_])
                    expr.append(prim)
                    for arg in reversed(prim.args):
                        stack.append((depth + 1, arg))
                elif len(pset.terminals[type_]) > 0:
                    term = random.choice(pset.terminals[type_])
                    if isclass(term):
                        term = term()
                    expr.append(term)
                else:
                    raise TypeError(
                        f"Invalid tree! No primitives or terminals of type {type_}"
                    )
    return expr


def get_values(ind: Any) -> List[float]:
    """
    Returns fitness values of a GP individual. Used by stats object
    """
    return ind.fitness.values

