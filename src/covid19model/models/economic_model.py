
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

from . import models


class EconomicModel():
    """
    A class to simulate the economic impact of pandemic mitigation
    =======================================================================================
    Params: 
    """

    def __init__(self,epidemologicalModel):
        # Provide acces to epidemological model parameters/variables
        self.epiModel = epidemologicalModel 
        # Extract population size of students, working, retired from a correctly initialized age-structured model
        # Student population taken as men and women below 20 y.o. 
        # Elderly population taken as men and women above 65 y.o.
        fullPop    = numpy.sum(self.epiModel.initN)
        studentPop = numpy.sum(self.epiModel.initN[0:3])
        retiredPop = numpy.sum(self.epiModel.initN[-3:])        
        # Load economic inputs
        SectoralData = pd.read_excel("../data/Sectoral_data.xlsx", sheet_name=None)
        Industry   = SectoralData["value added"].get("Industry breakdown")
        ValueAdded = SectoralData["value added"].Value * 1e6
        Employment = SectoralData["Employment"].Value * 1e3        
        # Raw economic data of 2018 from http://stat.nbb.be/index.aspx?DatasetCode=SUTAP38
        # Element [0] of each series is the aggregated value for all industries.
        Inputs = {'Industry breakdown'      : Industry,                 # Description of industry classes
                  'Value Added'             : ValueAdded,               # Economic value added by Belgian workers (per year)
                  'Employment'              : Employment,               # Number of employee per industry 
                  'Value Added per Employe' : ValueAdded / Employment,  # Value added per employee (per year)
                  'Employment fraction'     : Employment / Employment[0]}     # Ratio of employees in each industry without pandemics with respect to total number of workers.
        ReferenceNumbers = {'Population': fullPop,                                  # Full population of Belgium
                            'Student population' : studentPop,                      # Population of "students", i.e. Belgians below 20.
                            'Retired population': retiredPop,                       # Population of "retired", i.e. Belgians above 70.
                            'Working population' : Employment[0],                   # Population of "working" Belgians without pandemic.
                            'Non working population': fullPop - studentPop - retiredPop - Employment[0]}                   
        self.Inputs = Inputs
        self.ReferenceNumbers = ReferenceNumbers
        # Load adaptation to pandemic inputs according to a survey obtained on April 25th.
        # Element [0] of each series is the average value for all industries.
        StaffDistrib = pd.read_excel("../data/Staff distribution by sector.xlsx", sheet_name=None)
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

    def getEpidemologicalModelOutput(self,epidemologicalModel):
        return [epidemologicalModel.S, epidemologicalModel.E,
                        epidemologicalModel.I, epidemologicalModel.A,
                        epidemologicalModel.M, epidemologicalModel.Ctot,
                        epidemologicalModel.Mi, epidemologicalModel.ICU,
                        epidemologicalModel.R, epidemologicalModel.D,
                        epidemologicalModel.SQ, epidemologicalModel.EQ,
                        epidemologicalModel.IQ, epidemologicalModel.AQ,
                        epidemologicalModel.MQ, epidemologicalModel.RQ]

    def prepareMetaPopulationModel(self):

        # Load interaction matrices (hardcode for now but should be removed in the future)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        Nc_home = numpy.loadtxt("../data/Interaction_matrices/Belgium/BELhome.txt", dtype='f', delimiter='\t')
        Nc_work = numpy.loadtxt("../data/Interaction_matrices/Belgium/BELwork.txt", dtype='f', delimiter='\t')
        Nc_schools = numpy.loadtxt("../data/Interaction_matrices/Belgium/BELschools.txt", dtype='f', delimiter='\t')
        Nc_others = numpy.loadtxt("../data/Interaction_matrices/Belgium/BELothers.txt", dtype='f', delimiter='\t')
        Nc_all = numpy.loadtxt("../data/Interaction_matrices/Belgium/BELall.txt", dtype='f', delimiter='\t')
        initN = numpy.loadtxt("../data/Interaction_matrices/Belgium/BELagedist.txt", dtype='f', delimiter='\t')
        
        # Calculate priors
        # ~~~~~~~~~~~~~~~~
    
        # Calculate fraction of active population ( = individuals aged 20 to 65)
        activePopProb = (self.ReferenceNumbers['Working population']+self.ReferenceNumbers['Non working population'])/numpy.sum(self.epiModel.initN)        
        # Calculate what fraction of the "active" population has a job
        workingProb = self.ReferenceNumbers['Working population']/(self.ReferenceNumbers['Working population']+self.ReferenceNumbers['Non working population'])
        nonWorkingProb = 1 - workingProb

        # Define unemployed metapopulation 'manually'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        initN = []
        Nc = []
        initN.append(nonWorkingProb*activePopProb*self.epiModel.initN)
        Nc.append(Nc_home + Nc_schools + Nc_others)

        # Define working metapopulations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Ratio of employees in each industry without pandemics with respect to total number of workers.
        for i in range(len(self.Inputs['Employment fraction'][1:])): 
            initN.append(self.Inputs['Employment fraction'][i+1]*workingProb*activePopProb*self.epiModel.initN)
            Nc.append(Nc_home + Nc_work + Nc_schools + Nc_others)
        return initN,Nc

    def runMetaPopulationSimulation(self,T,checkpoints=None):
        # prepare initN and Nc for every metapopulation
        initN,Nc=self.prepareMetaPopulationModel()
        # make a copy of the model object
        mdl = self.epiModel
        # pre-allocation of results
        out=[]
        # loop over all metapopulations
        for i in range(len(initN)):
            # set initN and Nc
            mdl.initN = initN[i]
            mdl.Nc = Nc[i]
            # run model
            mdl.sim(T,checkpoints=checkpoints)
            # store result
            out.append(self.getEpidemologicalModelOutput(mdl))
        return out

    def calcAddedValue(self,out,confinement=False):
        VA = []
        VA_total = 0
        # loop over all metapopulations
        for i in range(len(out)):
            # Calculate fraction of active population that is working
            workingProb = self.ReferenceNumbers['Working population']/(self.ReferenceNumbers['Working population']+self.ReferenceNumbers['Non working population'])
            working = 0
            # loop over SEIR model states who are not 'sick'
            for j in [0,1,2,3,8]: 
                # Total number of 'healthy' workers in sector
                working = working + numpy.expand_dims((out[i][j].mean(axis=2))[4:-3,:].sum(axis=0)*workingProb,axis=0)
            # correct with confinement policy
            if confinement is True:
                fractionAtWorkPerSector = (self.Adaptation['Work at home'][i] + self.Adaptation['Mix home - office'][i] + self.Adaptation['Work at work'][i])/100
                working = fractionAtWorkPerSector*working
            # calculate the added value per day
            VA.append(working*self.Inputs['Value Added per Employe'][i]/365)
            VA_total = VA_total + VA[-1]
        return VA, VA_total

    #         for i in range(sum.shape[1]):
    #             sum[:,i] = sum[:,i]*fractionAtWorkPerSector
    #     # calculate value added per day
    #     VA_sector = numpy.zeros([sum.shape[0],sum.shape[1]])
    #     for i in range(sum.shape[1]):
    #         VA_sector[:,i] = sum[:,i]*self.Inputs['Value Added per Employe'][1:]/365


    # def calcWorkingPopulation(self,out):
    #     # Calculate fraction of active population that is working
    #     workingProb = self.ReferenceNumbers['Working population']/(self.ReferenceNumbers['Working population']+self.ReferenceNumbers['Non working population'])
    #     workPerSectorPerMetapopulation=[]
    #     for i in range(len(out)):
    #         # Sum over monte-carlo/repeats axis;
    #         # Extract for every population pool the active population (sum over index 4:-3) per day
    #         workPerSector=[]
    #         for j in range(len(out)):
    #             working = numpy.expand_dims((self.epiModelOut[j].mean(axis=2))[4:-3,:].sum(axis=0)*workingProb,axis=0) # 1xtN vector containing the number of people in given SEIR pool at work in one of the 37 economic sectors
    #             # reshape employment fraction per sector to 37x1 for matrix multiplication
    #             employFrac = numpy.expand_dims(self.Inputs["Employment fraction"][1:],axis=1)
    #             # matrix multiplication of employment fraction per sector and number of working individuals per timestep
    #             workPerSector.append(numpy.matmul(employFrac,working)) # 37xtN vector containing number of people in pool of SEIR at work in each sector
    #     workPerSectorPerMetapopulation.append(workPerSector)
    #     return workPerSectorPerMetapopulation

    # def calcAddedValue(self,epidemologicalModel,confinement=False):
    #     workPerSector = self.calcWorkingPopulation(epidemologicalModel)
    #     # Only S, E, I, A and R pools are at work, rest is sick at home/hospital: sum over all people at work
    #     sum = 0
    #     for i in [0,1,2,3,8]:
    #         sum = sum+workPerSector[i]
    #     # correct with confinement policy
    #     if confinement is True:
    #         fractionAtWorkPerSector = (self.Adaptation['Work at home'][1:] + self.Adaptation['Mix home - office'][1:] + self.Adaptation['Work at work'][1:])/100
    #         for i in range(sum.shape[1]):
    #             sum[:,i] = sum[:,i]*fractionAtWorkPerSector
    #     # calculate value added per day
    #     VA_sector = numpy.zeros([sum.shape[0],sum.shape[1]])
    #     for i in range(sum.shape[1]):
    #         VA_sector[:,i] = sum[:,i]*self.Inputs['Value Added per Employe'][1:]/365
    #     VA_total = numpy.sum(VA_sector,axis=0)

    #     return VA_sector,VA_total

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
        
    def ComputeValueAdded(self, StatusMatrix):
        # Returns an approximation of the value added to the economy by workers.
        # Hypothesis: 
        #   - Every worker produces in average the same amount of value added per day
        #   - No penalty is made for working from home instead of being physically on work place. 
        # Inputs to the function:
        #   SELF: The economic model object with data from Belgium Value Added per economical sector
        #   STATUSMATRIX: The matrix of occupation produced by ASSIGNOCCUPATION()
        # Output:
        #   The added value to the economy per day (in euros), according to the StatusMatrix.
    
        VA = self.Inputs['Value Added per Employe']
        TotalVA = 0
        
        for i in range(0,len(StatusMatrix[:,1])):
            if StatusMatrix[i,0] > 0:
                if StatusMatrix[i,1] < 4:
                    idx = StatusMatrix[i,0]
                    TotalVA = TotalVA + VA[idx]
    
        return(TotalVA / 365)