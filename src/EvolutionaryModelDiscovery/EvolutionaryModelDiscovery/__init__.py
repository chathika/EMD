'''EvolutionaryModelDiscovery: Automated agent rule generation and 
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
along with this program.  If not, see <http://www.gnu.org/licenses/>.'''

from datetime import time
from typing import Callable, List, Union
from pathlib import Path
import atexit
import importlib
import time

import nl4py
import pandas as pd

from .NetLogoWriter import NetLogoWriter
from .FactorGenerator import FactorGenerator
from .PrimitiveSetGenerator import PrimitiveSetGenerator
from .FactorImportances import FactorImportances
from .SimpleDEAPGP import *
from .Util import *


def exit_handler() -> None:
    remove_model_factors_file()
    #purge('.','.EMD.nlogo')

atexit.register(exit_handler)

class EvolutionaryModelDiscovery:
    
    def __init__(self, netlogo_path : str, model_path : str, setup_commands : List[str], 
                    measurement_reporters : List[str], ticks_to_run : int, 
                    go_command : str = 'go') -> None:
        """
        Evolutionary model discovery experiment. Can be used to perform genetic programming 
        of NetLogo models and factor importance analysis of resulting data using random 
        forest importance analysis. 

        See: 
            Gunaratne, C., & Garibay, I. (2020). Evolutionary model discovery 
            3of causal factors behind the socio-agricultural behavior of the 
            Ancestral Pueblo. Plos one, 15(12), e0239922.

            Gunaratne, C., Rand, W., & Garibay, I. (2021). Inferring mechanisms 
            of response prioritization on social media under information overload. 
            Scientific reports, 11(1), 1-12.

        :param netlogo_path: str path to folder with NetLogo executable
        :param model_path: str path to NetLogo .nlogo model file
        :param setup_commands: List[str] of NetLogo commands to be executed on simulation setup
        :param measurement_reporters: List[str] of NetLogo reporters to be run per simulation
                                            executed per simulation tick and reported to 
                                            objective function callback
        :param ticks_to_run: int number of ticks to run each simulation for
        :param go_command: str command to run NetLogo simulations (default: 'go')

        """
        # Initialize ABM
        self.model_init_data = {
            'model_path' : model_path,
            'setup_commands' : setup_commands,
            'measurement_commands' : measurement_reporters,
            'ticks_to_run' : ticks_to_run,
            'go_command' : go_command
        }
        self.replications = 1
        ModelFactors, netlogo_writer = self._parse_model_into_factors()
        # Starting NL4Py
        nl4py.initialize(netlogo_path)
        self.gp = SimpleDEAPGP(self.model_init_data, ModelFactors, netlogo_writer)
        self.factor_scores_file_name = 'FactorScores.csv'
    
    def set_mutation_rate(self, mutation_rate : float) -> None:
        self.gp.set_mutation_rate(mutation_rate)

    def set_crossover_rate(self, crossover_rate : float) -> None:
        self.gp.set_crossover_rate(crossover_rate)

    def set_generations(self, generations : int) -> None:
        self.gp.set_generations(generations)

    def set_replications(self, replications : int) -> None:
        self.replications = replications

    def set_population_size(self, population_size : int) -> None:
        self.gp.set_population_size(population_size)

    def set_objective_function(self, objective_function : Callable) -> None:
        self.gp.set_objective_function(objective_function)

    def set_depth(self, min : int, max : int) -> None:
        self.gp.set_depth(min,max)

    def set_factor_scores_file_name (self, name : str) -> None:
        self.factor_scores_file_name = str(Path(name))

    def set_is_minimize(self, is_minimize : bool) -> None:
        self.gp.set_is_minimize(is_minimize)
        
    def _parse_model_into_factors(self) -> 'EvolutionaryModelDiscovery.ModelFactors':
        """
        Parses NetLogo model and EMD annotations into Python classes 
        for syntax tree representation
        """

        netlogo_writer = NetLogoWriter(self.model_init_data['model_path'])
        #  Evolutionary Model Discovery Initializing
        #Reset Factors files
        create_model_factors_file()
        # Parsing NetLogo model into syntax tree primitives and Python classes
        # by reading in annotations from .nlogo file and generate EMD factors
        factor_generator = FactorGenerator()
        factor_generator.generate(netlogo_writer.get_factors_file_path())
        # Generate the ModelFactors.py file, Loading parsed primitives as Python classes
        primitive_set_generator = PrimitiveSetGenerator()
        primitive_set_generator.generate(factor_generator.get_factors(),netlogo_writer.get_EMD_return_type())
        module_name = get_model_factors_module_name()
        ModelFactors = importlib.import_module(f'EvolutionaryModelDiscovery.{module_name}')        
        return ModelFactors, netlogo_writer
    
    def evolve(self, num_procs : int = -1) -> pd.DataFrame:
        '''
        Conduct evolution using initialized genetic program

        :returns: pandas DataFrame with genetic program results
        '''
        # Begining evolution
        for run in range(self.replications):
            print('--- Starting GP Run {} ---'.format(run))             
            self.population, self.logbook, self.factor_scores = self.gp.evolve(num_procs=num_procs)
            self.factor_scores['Run'] = run
            for priority_col in ['Rule','Gen','Run']:
                col = self.factor_scores[priority_col]
                self.factor_scores.drop(labels=[priority_col], axis=1,inplace = True)
                self.factor_scores.insert(0, priority_col, col)
            if Path(self.factor_scores_file_name).is_file():
                old_factor_scores = pd.read_csv(self.factor_scores_file_name)
                old_factor_scores.append(self.factor_scores,ignore_index=True, 
                            sort=False).to_csv(self.factor_scores_file_name,index=False)
            else:
                self.factor_scores.to_csv(self.factor_scores_file_name,  header=True,index=False)
        print('--- Genetic program runs finished, output written to {} ---'.format(
                                                self.factor_scores_file_name))
        return self.factor_scores
        
    def get_factor_importances_calculator(self,
                                 factor_scores : Union[pd.DataFrame, str] = None) -> FactorImportances:
        """
        Returns FactorImportances object with trained Random Forest that can be used 
        to calculate Gini importance and permutation accuracy importance of factors.

        :param factor_scores: pandas dataframe or csv file location of factor scores generated by 
                                    genetic program.
        """
        try:
            if factor_scores is None:
                return FactorImportances(self.factor_scores)
            else:
                return FactorImportances(factor_scores)
        except AttributeError:
            raise Exception(('Either factor scores do not exist or GP not run. Do '
                                    'EvolutionaryModelDiscovery.evolve() first.'))

    @deprecated('Replaced with PEP 8 compliant variant {}'.format('set_mutation_rate'))
    def setMutationRate(self, mutationRate):
        self.set_mutation_rate(mutationRate)
    @deprecated('Replaced with PEP 8 compliant variant {}'.format('set_crossover_rate'))
    def setCrossoverRate(self, crossoverRate):
        self.set_crossover_rate(crossoverRate)
    @deprecated('Replaced with PEP 8 compliant variant {}'.format('set_generations'))
    def setGenerations(self, generations):
        self.set_generations(generations)
    @deprecated('Replaced with PEP 8 compliant variant {}'.format('set_replications'))
    def setReplications(self, replications):
        self.set_replications(replications)
    @deprecated('Replaced with PEP 8 compliant variant {}'.format('set_population_size'))
    def setPopulationSize(self, populationSize):
        self.set_population_size(populationSize)
    @deprecated('Replaced with PEP 8 compliant variant {}'.format('set_objective_function'))
    def setObjectiveFunction(self, objectiveFunction):
        self.set_objective_function(objectiveFunction)
    @deprecated('Replaced with PEP 8 compliant variant {}'.format('set_depth'))
    def setDepth(self, min, max):
        self.set_depth(min,max)
    @deprecated('Replaced with PEP 8 compliant variant {}'.format('set_factor_scores_file_name'))
    def setFactorScoresFileName (self, name):
        self.set_factor_scores_file_name(name)
    @deprecated('Replaced with PEP 8 compliant variant {}'.format('set_is_minimize'))
    def setIsMinimize(self, isMinimize ):
        self.set_is_minimize(isMinimize)



    
