# Copyright (c) 2022 by T.W. Alleman BIOMATH, Ghent University. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numba import jit
from covid19_DTM.models.jit_utils import jit_matmul_2D_1D, jit_matmul_2D_2D, jit_matmul_3D_2D, jit_matmul_klm_m, jit_matmul_klmn_n, matmul_q_2D
from pySODM.models.base import SDEModel
from .utils import stratify_beta_density, stratify_beta_regional, read_coordinates_place, construct_coordinates_Nc

# Ignore numba warnings
from numba.core.errors import NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

class COVID19_SEIQRD_hybrid_vacc_sto(SDEModel):
    """"""

    state_names = ['S', 'E', 'I', 'A', 'M_R', 'M_H', 'C_R', 'C_D', 'C_icurec','ICU_R', 'ICU_D', 'R', 'D', 'M_in', 'H_in','H_tot', 'Inf_in', 'Inf_out']
    parameter_names = ['beta', 'f_VOC', 'K_inf', 'K_hosp', 'sigma', 'omega', 'zeta','da', 'dm','dICUrec','dhospital', 'seasonality', 'N_vacc', 'e_i', 'e_s', 'e_h','Nc']
    parameter_stratified_names = [['s','a','h', 'c', 'm_C','m_ICU', 'dc_R', 'dc_D','dICU_R','dICU_D'],[]]
    dimension_names = ['age_groups','doses']

    @staticmethod
    def compute_rates(t, S, E, I, A, M_R, M_H, C_R, C_D, C_icurec, ICU_R, ICU_D, R, D, M_in, H_in, H_tot, Inf_in, Inf_out,
                        beta, f_VOC, K_inf, K_hosp, sigma, omega, zeta, da, dm,  dICUrec, dhospital, seasonality, N_vacc, e_i, e_s, e_h, Nc,
                        s, a, h, c, m_C, m_ICU, dc_R, dc_D, dICU_R, dICU_D):
        """
        Biomath extended SEIRD model for COVID-19
        *Stochastic implementation*
        """

        ###################
        ## Format inputs ##
        ###################

        # Extract fraction
        f_VOC = f_VOC[0,:]        
        # Prepend a 'one' in front of K_inf and K_hosp (cannot use np.insert with jit compilation)
        K_inf = np.array( ([1,] + list(K_inf)), np.float64)
        K_hosp = np.array( ([1,] + list(K_hosp)), np.float64)   
        
        #################################################
        ## Compute variant weighted-average properties ##
        #################################################

        # Hospitalization propensity (OR --> probability)
        h = (np.sum(f_VOC*K_hosp)*(h/(1-h)))/(1+ np.sum(f_VOC*K_hosp)*(h/(1-h)))

        # Latent period
        sigma = np.sum(f_VOC*sigma)
        # Vaccination
        e_i = jit_matmul_klm_m(e_i,f_VOC) # Reduces from (n_age, n_doses, n_VOCS) --> (n_age, n_doses)
        e_s = jit_matmul_klm_m(e_s,f_VOC)
        e_h = jit_matmul_klm_m(e_h,f_VOC)
        # Seasonality
        beta *= seasonality

        ####################################################
        ## Expand dims on first stratification axis (age) ##
        ####################################################

        a = np.expand_dims(a, axis=1)
        h = np.expand_dims(h, axis=1)
        c = np.expand_dims(c, axis=1)
        m_C = np.expand_dims(m_C, axis=1)
        m_ICU = np.expand_dims(m_ICU, axis=1)
        dc_R = np.expand_dims(dc_R, axis=1)
        dc_D = np.expand_dims(dc_D, axis=1)
        dICU_R = np.expand_dims(dICU_R, axis=1)
        dICU_D = np.expand_dims(dICU_D, axis=1)
        dICUrec = np.expand_dims(dICUrec, axis=1)
        h_acc = e_h*h

        ################################
        ## calculate total population ##
        ################################

        T = np.expand_dims(np.sum(S + E + I + A + M_R + M_H + C_R + C_D + C_icurec + ICU_R + ICU_D + R, axis=1),axis=1) # sum over doses

        ################################
        ## Compute the transitionings ##
        ################################

        # Compute infection pressure (IP) of all variants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        IP = np.expand_dims( np.sum( np.outer(beta*s*jit_matmul_2D_1D(Nc,np.sum(((I+A)/T)*e_i, axis=1)), f_VOC*K_inf), axis=1), axis=1)
        

        # Define the rates of the transitionings
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        size_dummy = np.ones(S.shape)

        rates = {
            'S': [IP*e_s,],
            'E': [size_dummy*(1/sigma),],
            'I': [size_dummy*(a/omega), size_dummy*(1-h_acc)*((1-a)/omega), size_dummy*h_acc*((1-a)/omega)],
            'A': [size_dummy*(1/da)],
            'M_R': [size_dummy*(1/dm)],
            'M_H': [size_dummy*(1/dhospital)*c*(1-m_C), size_dummy*(1/dhospital)*c*m_C, size_dummy*(1/dhospital)*(1-c)*(1-m_ICU), size_dummy*(1/dhospital)*(1-c)*m_ICU],
            'C_R': [size_dummy*(1/dc_R),],
            'C_D': [size_dummy*(1/dc_D),], 
            'ICU_R': [size_dummy*(1/dICU_R),],
            'ICU_D': [size_dummy*(1/dICU_D)],
            'C_icurec': [size_dummy*(1/dICUrec),],
            'R': [size_dummy*zeta,]
        }

        return rates

    @staticmethod
    def apply_transitionings(t, tau, transitionings, S, E, I, A, M_R, M_H, C_R, C_D, C_icurec, ICU_R, ICU_D, R, D, M_in, H_in, H_tot, Inf_in, Inf_out,
                            beta, f_VOC, K_inf, K_hosp, sigma, omega, zeta, da, dm,  dICUrec, dhospital, seasonality, N_vacc, e_i, e_s, e_h, Nc,
                            s, a, h, c, m_C, m_ICU, dc_R, dc_D, dICU_R, dICU_D):
        
        # Round the vaccination data
        N_vacc = tau*N_vacc

        ############################################
        ## Compute the vaccination transitionings ##
        ############################################

        dS = np.zeros(S.shape, np.float64)
        dR = np.zeros(R.shape, np.float64)

        # 0 --> 1 and  0 --> 2
        # ~~~~~~~~~~~~~~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,0] + R[:,0]
        VE = np.where(VE == 0, 1, VE)
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,0]/VE
        f_R = R[:,0]/VE
        # Compute transisitoning in zero syringes
        dS[:,0] = - (N_vacc[:,0] + N_vacc[:,2])*f_S 
        dR[:,0] = - (N_vacc[:,0]+ N_vacc[:,2])*f_R
        # Compute transitioning in one short circuit
        dS[:,1] =  N_vacc[:,0]*f_S # 0 --> 1 dose
        dR[:,1] =  N_vacc[:,0]*f_R 
        # Compute transitioning in two shot circuit
        dS[:,2] =  N_vacc[:,2]*f_S # 0 --> 2 doses
        dR[:,2] =  N_vacc[:,2]*f_R 

        # 1 --> 2 
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,1] + R[:,1]
        VE = np.where(VE == 0, 1, VE)
        # Compute fraction of VE in states S and R
        f_S = S[:,1]/VE
        f_R = R[:,1]/VE
        # Compute transitioning in partially vaccinated circuit
        dS[:,1] = dS[:,1] - N_vacc[:,1]*f_S
        dR[:,1] = dR[:,1] - N_vacc[:,1]*f_R
        # Compute transitioning in fully vaccinated circuit
        dS[:,2] = dS[:,2] + N_vacc[:,1]*f_S
        dR[:,2] = dR[:,2] + N_vacc[:,1]*f_R

        # 2 --> B
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,2] + R[:,2]
        VE = np.where(VE == 0, 1, VE)
        # Compute fraction of VE in states S and R
        f_S = S[:,2]/VE
        f_R = R[:,2]/VE
        # Compute transitioning in fully vaccinated circuit
        dS[:,2] = dS[:,2] - N_vacc[:,3]*f_S
        dR[:,2] = dR[:,2] - N_vacc[:,3]*f_R
        # Compute transitioning in boosted circuit
        dS[:,3] = dS[:,3] + N_vacc[:,3]*f_S
        dR[:,3] = dR[:,3] + N_vacc[:,3]*f_R

        # Update the system
        # ~~~~~~~~~~~~~~~~~

        # Flowchart states
        S_new = S + np.rint(dS) - transitionings['S'][0] + transitionings['R'][0]
        E_new = E + transitionings['S'][0] - transitionings['E'][0]
        I_new = I + transitionings['E'][0] - (transitionings['I'][0] + transitionings['I'][1] + transitionings['I'][2])
        A_new = A + transitionings['I'][0] - transitionings['A'][0]
        M_R_new = M_R + transitionings['I'][1] - transitionings['M_R'][0]
        M_H_new = M_H + transitionings['I'][2] - (transitionings['M_H'][0] + transitionings['M_H'][1] + transitionings['M_H'][2] + transitionings['M_H'][3])
        C_R_new = C_R + transitionings['M_H'][0] - transitionings['C_R'][0]
        C_D_new = C_D + transitionings['M_H'][1] - transitionings['C_D'][0]
        ICU_R_new = ICU_R + transitionings['M_H'][2] - transitionings['ICU_R'][0]
        ICU_D_new = ICU_D + transitionings['M_H'][3] - transitionings['ICU_D'][0]
        C_icurec_new = C_icurec + transitionings['ICU_R'][0] - transitionings['C_icurec'][0]
        R_new = R + np.rint(dR) + transitionings['A'][0] + transitionings['M_R'][0] + transitionings['C_R'][0] + transitionings['C_icurec'][0] - transitionings['R'][0]
        D_new = D + transitionings['ICU_D'][0] + transitionings['C_D'][0]

        # Derivative states
        M_in_new =  (transitionings['I'][1] + transitionings['I'][2])/tau
        H_in_new = (transitionings['M_H'][0] + transitionings['M_H'][1] + transitionings['M_H'][2] + transitionings['M_H'][3])/tau
        H_out_new = (transitionings['C_R'][0] + transitionings['C_icurec'][0] + transitionings['ICU_D'][0] + transitionings['C_D'][0])/tau
        H_tot_new = H_tot + (H_in_new - H_out_new)*tau
        Inf_in_new = transitionings['S'][0]/tau
        Inf_out_new = (transitionings['A'][0] + transitionings['M_R'][0])/tau

        # Make absolutely sure the vaccinations don't push the S or R states below zero
        S_new = np.where(S_new < 0, 0, S_new)
        R_new = np.where(R_new < 0, 0, R_new)

        return (S_new, E_new, I_new, A_new, M_R_new, M_H_new, C_R_new, C_D_new, C_icurec_new, ICU_R_new, ICU_D_new, R_new, D_new, M_in_new, H_in_new, H_tot_new, Inf_in_new, Inf_out_new)

class COVID19_SEIQRD_spatial_hybrid_vacc_sto(SDEModel):

    # ...state variables and parameters
    state_names = ['S', 'S_work', 'E', 'I', 'A', 'M_R', 'M_H', 'C_R', 'C_D', 'C_icurec','ICU_R', 'ICU_D', 'R', 'D', 'M_in', 'H_in','H_tot']
    parameter_names = ['beta_R', 'beta_U', 'beta_M', 'f_VOC', 'K_inf', 'K_hosp', 'sigma', 'omega', 'zeta','da', 'dm','dICUrec','dhospital', 'seasonality', 'N_vacc', 'e_i', 'e_s', 'e_h', 'Nc', 'Nc_work', 'NIS']
    parameter_stratified_names = [['area', 'p'],['s','a','h', 'c', 'm_C','m_ICU', 'dc_R', 'dc_D','dICU_R','dICU_D'],[]]
    dimension_names = ['NIS','age_groups','doses']

    @staticmethod
    def compute_rates(t, S, S_work, E, I, A, M_R, M_H, C_R, C_D, C_icurec, ICU_R, ICU_D, R, D, M_in, H_in, H_tot, # time + SEIRD classes
                  beta_R, beta_U, beta_M, f_VOC, K_inf, K_hosp, sigma, omega, zeta, da, dm, dICUrec, dhospital, seasonality, N_vacc, e_i, e_s, e_h, Nc, Nc_work, NIS, # SEIRD parameters
                  area, p, # spatially stratified parameters. 
                  s, a, h, c, m_C, m_ICU, dc_R, dc_D, dICU_R, dICU_D):

        ###################
        ## Format inputs ##
        ###################

        f_VOC = f_VOC[0,:]        
        # Prepend a 'one' in front of K_inf and K_hosp (cannot use np.insert with jit compilation)
        K_inf = np.array( ([1,] + list(K_inf)), np.float64)
        K_hosp = np.array( ([1,] + list(K_hosp)), np.float64)   

        #################################################
        ## Compute variant weighted-average properties ##
        #################################################

        # Hospitalization propensity
        h = (np.sum(f_VOC*K_hosp)*(h/(1-h)))/(1+ np.sum(f_VOC*K_hosp)*(h/(1-h)))
        # Latent period
        sigma = np.sum(f_VOC*sigma)
        # Vaccination
        e_i = jit_matmul_klmn_n(e_i,f_VOC) # Reduces from (n_NIS, n_age, n_doses, n_VOCS) --> (n_NIS, n_age, n_doses)
        e_s = jit_matmul_klmn_n(e_s,f_VOC)
        e_h = jit_matmul_klmn_n(e_h,f_VOC)
        # Seasonality
        beta_R *= seasonality
        beta_U *= seasonality
        beta_M *= seasonality

        ####################################################
        ## Expand dims on first stratification axis (age) ##
        ####################################################

        a = np.expand_dims(a, axis=1)
        h = np.expand_dims(h, axis=1)
        c = np.expand_dims(c, axis=1)
        m_C = np.expand_dims(m_C, axis=1)
        m_ICU = np.expand_dims(m_ICU, axis=1)
        dc_R = np.expand_dims(dc_R, axis=1)
        dc_D = np.expand_dims(dc_D, axis=1)
        dICU_R = np.expand_dims(dICU_R, axis=1)
        dICU_D = np.expand_dims(dICU_D, axis=1)
        dICUrec = np.expand_dims(dICUrec, axis=1)
        h_acc = e_h*h

        ################################
        ## calculate total population ##
        ################################

        T = np.sum(S + E + I + A + M_R + M_H + C_R + C_D + C_icurec + ICU_R + ICU_D + R, axis=2) # Sum over doses

        ################################
        ## Compute infection pressure ##
        ################################

        # For total population and for the relevant compartments I and A
        G = S.shape[0] # spatial stratification
        N = S.shape[1] # age stratification
        D = S.shape[2] # dose stratification

        # Define effective mobility matrix place_eff from user-defined parameter p[patch]
        place_eff = np.outer(p, p)*NIS + np.identity(G)*(NIS @ (1-np.outer(p,p)))
        
        # Expand beta to size G
        beta = stratify_beta_regional(beta_R, beta_U, beta_M, G)*np.sum(f_VOC*K_inf)

        # Compute populations after application of 'place' to obtain the S, I and A populations
        T_work = np.expand_dims(np.transpose(place_eff) @ T, axis=2)
        I_work = matmul_q_2D(np.transpose(place_eff), I)
        A_work = matmul_q_2D(np.transpose(place_eff), A)
        # The following line of code is the numpy equivalent of the above loop (verified)
        #S_work = np.transpose(np.matmul(np.transpose(S_post_vacc), place_eff))

        # Compute infectious work population (11,10)
        infpop_work = np.sum( (I_work + A_work)/T_work*e_i, axis=2)
        infpop_rest = np.sum( (I + A)/np.expand_dims(T, axis=2)*e_i, axis=2)

        # Multiply with number of contacts
        multip_work = np.expand_dims(jit_matmul_3D_2D(Nc - Nc_work, infpop_work), axis=2) # All contacts minus home contacts on visited patch
        multip_rest = np.expand_dims(jit_matmul_3D_2D(Nc_work, infpop_rest), axis=2) # Home contacts always on home patch

        # Multiply result with beta
        multip_work *= np.expand_dims(np.expand_dims(beta, axis=1), axis=2)
        multip_rest *= np.expand_dims(np.expand_dims(beta, axis=1), axis=2)

        ################################
        ## Compute the transitionings ##
        ################################

        # Define the rates of the transitionings
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        size_dummy = size_dummy = np.ones([G,N,D], np.float64)
        rates = {
            'S': [multip_rest*e_s,],
            'S_work': [multip_work*e_s,],
            'E': [size_dummy*(1/sigma),],
            'I': [np.squeeze(a/omega)[np.newaxis, :, np.newaxis]*size_dummy, (1-h_acc)*((1-a)/omega), h_acc*((1-a)/omega)],
            'A': [size_dummy*(1/da),],
            'M_R': [size_dummy*(1/dm),],
            'M_H': [np.squeeze((1/dhospital)*c*(1-m_C))[np.newaxis, :, np.newaxis]*size_dummy,
                    np.squeeze((1/dhospital)*c*m_C)[np.newaxis, :, np.newaxis]*size_dummy,
                    np.squeeze((1/dhospital)*(1-c)*(1-m_ICU))[np.newaxis, :, np.newaxis]*size_dummy,
                    np.squeeze((1/dhospital)*(1-c)*m_ICU)[np.newaxis, :, np.newaxis]*size_dummy],
            'C_R': [np.squeeze(1/dc_R)[np.newaxis, :, np.newaxis]*size_dummy,],
            'C_D': [np.squeeze(1/dc_D)[np.newaxis, :, np.newaxis]*size_dummy,], 
            'ICU_R': [np.squeeze(1/dICU_R)[np.newaxis, :, np.newaxis]*size_dummy,],
            'ICU_D': [np.squeeze(1/dICU_D)[np.newaxis, :, np.newaxis]*size_dummy,],
            'C_icurec': [np.squeeze(1/dICUrec)[np.newaxis, :, np.newaxis]*size_dummy,],
            'R': [size_dummy*zeta,],
        }

        return rates
    
    @staticmethod
    def apply_transitionings(t, tau, transitionings, S, S_work, E, I, A, M_R, M_H, C_R, C_D, C_icurec, ICU_R, ICU_D, R, D, M_in, H_in, H_tot, # time + SEIRD classes
                             beta_R, beta_U, beta_M, f_VOC, K_inf, K_hosp, sigma, omega, zeta, da, dm, dICUrec, dhospital, seasonality, N_vacc, e_i, e_s, e_h, Nc, Nc_work, NIS, # SEIRD parameters
                             area, p, # spatially stratified parameters. 
                             s, a, h, c, m_C, m_ICU, dc_R, dc_D, dICU_R, dICU_D):

        ############################
        ## Update work population ##
        ############################

        place_eff = np.outer(p, p)*NIS + np.identity(S.shape[0])*(NIS @ (1-np.outer(p,p)))
        S_work_new = matmul_q_2D(np.transpose(place_eff), S)

        ############################################
        ## Compute the vaccination transitionings ##
        ############################################

        # Round the vaccination data
        N_vacc = tau*N_vacc

        dS = np.zeros(S.shape, np.float64)
        dR = np.zeros(R.shape, np.float64)

        # 0 --> 1 and  0 --> 2
        # ~~~~~~~~~~~~~~~~~~~~
        # Compute vaccine eligible population
        VE = S[:,:,0] + R[:,:,0]
        VE = np.where(VE == 0, 1, VE)
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,0]/VE
        f_R = R[:,:,0]/VE
        # Compute transisitoning in zero syringes
        dS[:,:,0] = - (N_vacc[:,:,0] + N_vacc[:,:,2])*f_S
        dR[:,:,0] = - (N_vacc[:,:,0]+ N_vacc[:,:,2])*f_R
        # Compute transitioning in one short circuit
        dS[:,:,1] =  N_vacc[:,:,0]*f_S # 0 --> 1 dose
        dR[:,:,1] =  N_vacc[:,:,0]*f_R 
        # Compute transitioning in two shot circuit
        dS[:,:,2] =  N_vacc[:,:,2]*f_S # 0 --> 2 doses
        dR[:,:,2] =  N_vacc[:,:,2]*f_R

        # 1 --> 2 
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,:,1] + R[:,:,1]
        VE = np.where(VE == 0, 1, VE)
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,1]/VE
        f_R = R[:,:,1]/VE
        # Compute transitioning in one short circuit
        dS[:,:,1] = dS[:,:,1] - N_vacc[:,:,1]*f_S
        dR[:,:,1] = dR[:,:,1] - N_vacc[:,:,1]*f_R
        # Compute transitioning in two shot circuit
        dS[:,:,2] = dS[:,:,2] + N_vacc[:,:,1]*f_S
        dR[:,:,2] = dR[:,:,2] + N_vacc[:,:,1]*f_R


        # 2 --> B
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,:,2] + R[:,:,2]
        VE = np.where(VE == 0, 1, VE)
        # Compute fraction of VE in states S and R
        f_S = S[:,:,2]/VE
        f_R = R[:,:,2]/VE
        # Compute transitioning in fully vaccinated circuit
        dS[:,:,2] = dS[:,:,2] - N_vacc[:,:,3]*f_S
        dR[:,:,2] = dR[:,:,2] - N_vacc[:,:,3]*f_R
        # Compute transitioning in boosted circuit
        dS[:,:,3] = dS[:,:,3] + N_vacc[:,:,3]*f_S
        dR[:,:,3] = dR[:,:,3] + N_vacc[:,:,3]*f_R


        # Update the system
        # ~~~~~~~~~~~~~~~~~

        # Flowchart states
        S_new = S + np.rint(dS) - transitionings['S'][0] - transitionings['S_work'][0] + transitionings['R'][0]
        E_new = E + transitionings['S'][0] + transitionings['S_work'][0] - transitionings['E'][0]
        I_new = I + transitionings['E'][0] - (transitionings['I'][0] + transitionings['I'][1] + transitionings['I'][2])
        A_new = A + transitionings['I'][0] - transitionings['A'][0]
        M_R_new = M_R + transitionings['I'][1] - transitionings['M_R'][0]
        M_H_new = M_H + transitionings['I'][2] - (transitionings['M_H'][0] + transitionings['M_H'][1] + transitionings['M_H'][2] + transitionings['M_H'][3])
        C_R_new = C_R + transitionings['M_H'][0] - transitionings['C_R'][0]
        C_D_new = C_D + transitionings['M_H'][1] - transitionings['C_D'][0]
        ICU_R_new = ICU_R + transitionings['M_H'][2] - transitionings['ICU_R'][0]
        ICU_D_new = ICU_D + transitionings['M_H'][3] - transitionings['ICU_D'][0]
        C_icurec_new = C_icurec + transitionings['ICU_R'][0] - transitionings['C_icurec'][0]
        R_new = R + np.rint(dR) + transitionings['A'][0] + transitionings['M_R'][0] + transitionings['C_R'][0] + transitionings['C_icurec'][0] - transitionings['R'][0]
        D_new = D + transitionings['ICU_D'][0] + transitionings['C_D'][0]

        # Derivative states
        M_in_new =  (transitionings['I'][1] + transitionings['I'][2])/tau
        H_in_new = (transitionings['M_H'][0] + transitionings['M_H'][1] + transitionings['M_H'][2] + transitionings['M_H'][3])/tau
        H_out_new = (transitionings['C_R'][0] + transitionings['C_icurec'][0] + transitionings['ICU_D'][0] + transitionings['C_D'][0])/tau
        H_tot_new = H_tot + (H_in_new - H_out_new)*tau

        # Make absolutely sure the vaccinations don't push the S or R states below zero
        S_new = np.where(S_new < 0, 0, S_new)
        R_new = np.where(R_new < 0, 0, R_new)

        return (S_new, S_work_new, E_new, I_new, A_new, M_R_new, M_H_new, C_R_new, C_D_new, C_icurec_new, ICU_R_new, ICU_D_new, R_new, D_new, M_in_new, H_in_new, H_tot_new)
