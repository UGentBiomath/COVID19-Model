
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as networkx
import numpy as numpy
import scipy as scipy
import scipy.integrate
import pandas as pd
from random import choices
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from scipy import interpolate as inter
import copy
import models

class EconomicModel():
    """
    A class to simulate the economic impact of pandemic mitigation
    =======================================================================================
    Params: 
    """
      
        
    def LoadInputs(self):
        # Load economic inputs
        SectoralData = pd.read_excel("..\\data\\Sectoral_data.xlsx", sheet_name=None)
        
        Industry   = SectoralData["value added"].get("Industry breakdown")
        ValueAdded = SectoralData["value added"].Value * 1e6
        Employment = SectoralData["Employment"].Value * 1e3
    
        
        
        # Taken from https://statbel.fgov.be/en/themes/population/structure-population.
        # Student population taken as men and women below 20 y.o. 
        # Elderly population taken as men and women above 70 y.o.
        FullPop    = 11431406
        StudentPop =  2561879
        RetiredPop   =  1548845


        # Raw economic data of 2018 from http://stat.nbb.be/index.aspx?DatasetCode=SUTAP38
        # Element [0] of each series is the aggregated value for all industries.
        Inputs = {'Industry breakdown'      : Industry,                 # Description of industry classes
                  'Value Added'             : ValueAdded,               # Economic value added by Belgian workers (per year)
                  'Employment'              : Employment,               # Number of employee per industry
                  'Value Added per Employe' : ValueAdded / Employment,  # Value added per employee
                  'Employment fraction'     : Employment / Employment[0]}     # Ratio of employees in each industry without pandemics with respect to total number of workers.

        ReferenceNumbers = {'Population': FullPop,                                  # Full population of Belgium
                            'Student population' : StudentPop,                      # Population of "students", i.e. Belgians below 20.
                            'Retired population': RetiredPop,                       # Population of "retired", i.e. Belgians above 70.
                            'Working population' : Employment[0]}                   # Population of "working" Belglian without pandemic.
        
        self.Inputs = Inputs
        self.ReferenceNumbers = ReferenceNumbers
        
        # Load adaptation to pandemic inputs according to a survey obtained on April 25th.
        # Element [0] of each series is the average value for all industries.
        StaffDistrib = pd.read_excel("..\\data\\Staff distribution by sector.xlsx", sheet_name=None)
        WorkAtHome = StaffDistrib["Formated data"].get("telework")
        Mix        = StaffDistrib["Formated data"].get("mix telework-workplace")
        WorkAtWork = StaffDistrib["Formated data"].get("at workplace")
        Unemployed = StaffDistrib["Formated data"].get("temporary unemployed")
        Absent     = StaffDistrib["Formated data"].get("absent")

        Adaptation = {'Work at home'            : WorkAtHome,
                      'Mix home - office'       : Mix,
                      'Work at work'            : WorkAtWork,
                      'Temporary Unemployed'    : Unemployed,
                      'Absent'                  : Absent,
                      'Date of Survey'          : 'April 25th'}        
        
        self.Adaptation = Adaptation
        return(self)
    
    def AssignOccupation(self, N, ConfinementPolicy=True):
        # Assign a random occupation by assuming random age and, if between 20 and 70, random occupation.
        #   N is the number of occupation to assign.
        #   ConfinementPolicy : True  = maximal confinement. Non-working people are confined and working people 
        #                               are in the industry-dependant state of April 25th (maximal telework and 
        #                               mix, minimal work at work).
        #                       False = minimal confinement. Non-working people are not confined and working people 
        #                               work at work (office/factory/etc.)
        #   The returned variable is a numpy matrix containing the occupation information. Each line refers to 
        #   one individual.
        #       Firs column: index of occupation
        #           -2 : student
        #           -1 : retired
        #            0 : non-working
        #            1 - 37 : Occupation in one of the 37 industries.
        #       Second column: Working state
        #           -1 : Confined (children, retired, non-working individuals)
        #            0 : Unconfined (children, retired, non-working individuals)
        #            1 : Work at Home
        #            2 : Mix of Work at Home and Work at Work (office/factory/etc.)
        #            3 : Work at Work (office/factory/etc.)
        #            4 : Temporarily unemployed
        #            5 : Absent 

        StudentProb = self.ReferenceNumbers['Student population'] / self.ReferenceNumbers['Population']
        WorkingProb = self.ReferenceNumbers['Working population'] / self.ReferenceNumbers['Population']
        RetiredProb = self.ReferenceNumbers['Retired population'] / self.ReferenceNumbers['Population']
        NonWorkingProb = 1 - StudentProb - WorkingProb - RetiredProb

        status = [i for i in range(-2,38)]
        weights = [StudentProb, RetiredProb, NonWorkingProb];
        w2 = self.Inputs['Employment fraction'][1:].tolist() 

        for x in w2:
            weights.append(x * WorkingProb.tolist()) 
        
        StatusVector = choices(population=status,
                               weights=weights,
                               k=N)
        
        WorkingState = StatusVector.copy()
        w_work = numpy.array( [self.Adaptation["Work at home"], 
                               self.Adaptation["Mix home - office"], 
                               self.Adaptation["Work at work"], 
                               self.Adaptation["Temporary Unemployed"],
                               self.Adaptation["Absent"]])
        
        self.WorkingState_weights = w_work;
        
        if ConfinementPolicy : 
            for i in range(len(WorkingState)):
                if StatusVector[i] <= 0:
                    WorkingState[i] = -1
                else :
                    WorkingState[i] = choices(population=[pop for pop in range(1,6)],         
                                              weights= w_work[:,StatusVector[i]],
                                              k=1)[0]
        else:
            for i in range(len(WorkingState)):
                if StatusVector[i] <= 0:
                    WorkingState[i] = 0
                else :
                    WorkingState[i] = 3         
            
                
        FullStatus = numpy.array([StatusVector, 
                                 WorkingState])
        
        return(FullStatus.transpose())

    def ChangeConfinementPolicy(self,StatusMatrix, SectorClass, NewPolicy):
        # Allows to change the confinement policy for selected sector classes 
        # StatusMatrix is a matrix containing the full status of N individuals, 
        #   with one individual per line. See function AssignOccupation for the 
        #   definition of an individual.
        # SectorClass is the index of the selected sector. Must be between -2 and 37.
        # NewPolicy is between 0 (full confinement) and 1 (no confinement). 
        #   - All adaptations (work from home, mix, temporary unemployed, absent) 
        #   vary linearly by a factor (1 - NewPolicy). The work from work 
        #   (office/factory/etc) varies linearly with the factor NewPolicy.
        #   - Non-working population can only be Confined (NewPolicy = 0) or 
        #   unconfined (NewPolicy = 1)
        #
        # An updated StatusMatrix is returned
                
        if SectorClass > 0:
            w_work = self.WorkingState_weights[:,SectorClass].copy()
            for i in [0, 1, 3, 4]:
                w_work[i] = w_work[i] * (1 - NewPolicy)
            
            w_work[2] = w_work[2]  + (1 - w_work[2]) * NewPolicy
            
        for i in range(0,len(StatusMatrix[:,1])):
            if StatusMatrix[i,0] == SectorClass:
                if StatusMatrix[i,0] <= 0:
                    StatusMatrix[i,1] = NewPolicy - 1
                else :
                    StatusMatrix[i,1] = choices(population=[pop for pop in range(1,6)],         
                                                weights= w_work,
                                                k=1)[0]
                        
        return(StatusMatrix)
    

    def ComputeOccupation(self,StatusMatrix):
        # Returns statistics on occupation.
        # - Number of unconfined non-workers (students + retired + non-working individuals)
        # - Number of working workers (at home, at the office/factory/etc, mix)

        Confined = 0
        Unconfined = 0
        Working = 0
        Unemployed = 0
        
        for i in range(0,len(StatusMatrix[:,1])):
            if StatusMatrix[i,0] <= 0:
                if StatusMatrix[i,1] == -1:
                    Confined = Confined + 1
                else:
                    Unconfined = Unconfined + 1
            else:
                if StatusMatrix[i,1] < 4:
                    Working = Working + 1
                else:
                    Unemployed = Unemployed + 1
        
        Results = {'Confined' : Confined,
                   'Unconfined' : Unconfined,
                   'Working' : Working,
                   'Unemployed' : Unemployed}
            
        return(Results)
        
#    def ComputeEconomicAddedValue(self, model):
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    