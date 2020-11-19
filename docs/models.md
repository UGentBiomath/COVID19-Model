## Introduction

After an initial outbreak in early 2020 in Wuhan, China, Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) has spread globally. SARS-CoV-2 is capable of sustained human-to-human trans-
mission, and may cause severe disease, especially in older individuals. The COVID-19 pandemic has, in general, shown a remarkedly low incidence among children and young adults. Children have a lower susceptibility to infection and a lower propensity to show clinical symptoms. Furthermore, pre-symptomatic transmission is a major contributor to SARS-CoV-2 spread.
Unfortunately, pharmaceutical interventions such as vaccination and antiviral drugs are not yet available. On March 13th, 2020, the Belgian governments imposed strict social restrictions after tracing methods had failed to prevent large-scale spread of SARS-CoV-2. One and a half months later, the curve was succesfully flattened and social restrictions were gradually relaxed during the months of May, June and the beginning of July. In spite of these relaxations, hospitalizations kept declining. It is expected that during the coming year(s), preventing COVID-19 outbreaks will depend mostly on the successful implementation of non-pharmaceutical interventions such as social distancing, testing, contact tracing and quarantine. Hence the need for well-informed models that can assist policymakers in choosing the best non-pharmaceutical interventions in case of another large-scale SARS-CoV-2 outbreak. Currently, two other models exist to make predictions for Belgium, the agent-based model of Willem et al. and the discrete-time, stochastic metapopulation model of Coletti et al.

We built a deterministic, continuous-time, age-stratified, extended SEIRD model and calibrated it to national Belgian hospitalization data. The model accounts for pre-symptomatic and asymptomatic transmission. Furthermore, the susceptibility to SARS-CoV-2, the severity of the disease and the susceptibility to a subclinical infection depend on the age of the individual. We used social contact rates from a 2012 study by Willem to model age-specific social mixing. Because detailed hospitalization data are not made publicly available by the Belgian Scientific Institute of Public Health (Sciensano), we derived age-stratified hospitalization parameters from data of 370 patients treated in two Ghent (Belgium) hospitals. To overcome this unconstructive data policy, we used the computed hospitalization parameters as temporary proxies to fit the model to the total number of patients in Belgian hospitals and ICU units. Using the model, we computed the basic reproduction number ($R_0$) during the March 2020 epidemic. Because the model systematically overestimates the hospitalizations during lockdown relaxation, we re-estimated the basic reproduction using data from May 2020 until July 2020.

## The model

### Model dynamics

The SEIR(D) model was first proposed in 1929 by two Scottish scientists. It is a compartmental model that subdivides the human population into four groups: 1) susceptible individuals, 2) exposed individuals in the latent phase 1, 3) infectious individuals capable of transmitting the disease and 4) individuals removed from the population either through immunization or death. Despite being a simple and idealized reality, the SEIR(D) dynamics are used extensively to predict the outbreak of infectious diseases and this was no different during the SARS-CoV-2 outbreak earlier this year.

In this work, we extend the SEIRD model to incorporate more expert knowledge on SARS-CoV-2. For that purpose, the infectious compartment is split into four parts. The first is a period of presymptomatic infectiousness because several studies have shown that pre-symptomatic transmission is a dominant transmission mechanism of SARS-CoV-2. After the period of pre-symptomatic transmission, three possible infectious outcomes are modeled: (1) Asymptomatic outcome, for individuals who show no symptoms at all, (2) Mild outcome, for individuals with mild symptoms who recover at home, and (3) Hospitalization, when mild symptoms worsen. Children and young adults have a high propensity to experience an asymptomatic or mild outcome, while older individual have a high propensity to be hospitalized.

In general, Belgian hospitals have two wards for COVID-19 patients: 1) Cohort, where patients are not monitored permanently and 2) Intensive care, for patients with the most severe symptoms. Intensive care includes permanent monitoring, the use of ventilators or the use of extracorporeal membrane oxygenation (ECMO). Patients generally spend limited time in the emergency room and/or in a buffer ward before going to Cohort. After spending limited time in Cohort, some patients are transferred to ICU. Patients can perish in both wards, but mortalities are generally lower in Cohort. After a stay in an ICU, patients return to Cohort for recovery in the hospital. During the recovery stay, mortality is limited. The above is a short summary of hospital dynamics based on interviewing Ghent University hospital staff and examining the hospital data.

<p align="center">
<img src="_static/figs/flowchart_full.png" alt="drawing" width="600"/>

<em> Extended SEIRD dynamics of the BIOMATH COVID-19 model. Nodes represent model states, edges denote transfers. An overview of the model parameters can be found on the bottom of this page.</em>
</p>

### Model framework and equations

#### Age-stratification
We introduced heterogeneity in the deterministic implementation by means of age-stratification. Every population compartment is split into a number of age classes, the age-groups have different contact rates with other age-groups and the disease progresses differently for each age-group, making the model behaviour more realistic. Our age-stratified model consists of 9 age classes, i.e., [0-10[, [10-20[, [20-30[, [30-40[, [40-50[, [50-60[, [60-70[, [70-80[, [80- $\infty$[. The age-stratified implementation provides a good balance between added complexity and computational resources.

#### Deterministic framework

Our extended SEIRD model is implemented using two frameworks: a deterministic and a stochastic framework. The deterministic equations are obtained by writing down the following equation,

`rate of change = in - out`,

for every of the 11 population compartments. This results in the following system of coupled ordinary differential equations,

$$
\begin{eqnarray}
\dot{S_i} &=& - \beta s_i S_i \sum_{j=1}^{N} N_{c,ij}  \Big( \frac{I_j+A_j}{T_j} \Big) + \zeta R_i, \\
\dot{E_i} &=& \beta s_i  S_i \sum_{j=1}^{N} N_{c,ij} \Big( \frac{I_j+A_j}{T_j} \Big) - (1/\sigma) \cdot E_i,  \\
\dot{I_i} &=& (1/\sigma) E_i - (1/\omega) I_i, \\
\dot{A_i} &=& (\text{a}_i/\omega) I_i - (1/d_{\text{a}}) A_i, \\
\dot{M_i} &=&  ((1-\text{a}_i) / \omega ) I_i - ( (1-h_i)/d_m + h_i/d_{\text{hospital}} ) M_i, \\
\dot{ER_i} &=& (h_i/d_{\text{hospital}}) M_i - (1/d_{\text{ER}}) ER_i, \\
\dot{C_i} &=& c_i (1/d_{\text{ER}}) ER_i  - (m_{C, i}/d_{c,D}) C_i - ((1 - m_{C, i})/d_{c,R}) C_i, \\
\dot{ICU_i} &=& (1-c_i) (1/d_{\text{ER}}) ER_i - (m_{ICU,i}/d_{\text{ICU},D}) ICU_i  \\
&& - ((1-m_{ICU,i})/d_{\text{ICU},R}) ICU_i,\\
\dot{C}_{\text{ICU,rec,i}} &=& ((1-m_{ICU,i})/d_{\text{ICU},R}) ICU_i - (1/d_{\text{ICU,rec}}) C_{\text{ICU,rec,i}}, \\
\dot{D_i} &=&  (m_{ICU,i}/d_{\text{ICU},D}) ICU_i +  (m_{C,i}/d_{\text{c},D}) C_i , \\
\dot{R_i} &=&  (1/d_a) A_i + ((1-h_i)/d_m) M_i + ((1-m_{C,i})/d_{c,R}) C_i \\
&& + (1/d_{\text{ICU,rec}}) C_{\text{ICU,rec,i}} - \zeta R_i,
\end{eqnarray}
$$

for $i = 1,2,...,9$. Here, $T_i$ stands for total population, $S_i$ stands for susceptible, $E_i$ for exposed, $I_i$ for pre-symptomatic and infectious, $A_i$ for asymptomatic and infectious, $M_i$ for mildly symptomatic and infectious, $ER_i$ for emergency room and/or buffer ward, $C_i$ for cohort, $C_{\text{ICU,rec,i}}$ for a recovery stay in Cohort coming from Intensive Care, $ICU_i$ for Intensive Care Unit, $D_i$ for dead and $R_i$ for recovered. Using the above notation, all model states are 9x1 vectors,

$$
\begin{equation}
     \mathbf{S} = [S_1(t)\ S_2(t)\ ...\ S_i(t)]^T,
\end{equation}
$$

where $S_i(t)$ denotes the number of susceptibles in age-class i at time t after the introduction of the virus in the population.

These equations are implemented in the function `COVID19_SEIRD` located in `src/covid19model/models.py`. The integration is performed in `_sim_single` located in `src/covid19model/base.py` by using Scipy's `solve_ivp`. The integrator timestep depends on the rate of change of the system and the solver method is thus referred to as a 'continuous-time' solver. The implementation uses non-integer individuals.

#### Stochastic framework

By defining the probabilities of transitioning (propensities) from one state to another, a system of coupled stochastic difference equations (SDEs) can be obtained. The probability to transition from one state to another is assumed to be exponentially distributed. As an example, consider the average time a patient spends in an ICU when recovering, which is $d_{\text{ICU,R}} = 9.9$ days. The chances of survival in ICU are $(1-m_{\text{ICU,i}})$, where $m_{\text{ICU,i}}$ is the mortality in ICU for an individual in age group $i$. The probability of transitioning from state ICU to state $C_{\text{ICU,rec}}$ on any given day and for an individual in age group $i$ is,

$$
\begin{equation}
P(ICU_i \rightarrow C_{\text{ICU,rec,i}}) = 1 - \text{exp} \Bigg[ - \frac{1-m_{\text{ICU},i}}{d_{\text{ICU,R}}}\Bigg].
\end{equation}
$$

If a transitioning between states is defined as "succes", we can regard the number of individuals transitioning from ICU to a Cohort recovery ward as a binomial experiment. On a given day, the number of individuals transitioning is,

$$
\begin{equation}
(\text{ICU}_i \rightarrow C_{\text{ICU,rec,i}})(k) \sim \text{Binomial}\Bigg(\text{ICU}_i(k),\ 1 - \text{exp}\Bigg[- \frac{1-m_{\text{ICU,i}}}{d_{\text{ICU,R}}}\Bigg]\Bigg).
\end{equation}
$$

For a discrete stepsize $l$, there are 15 possible transitions,

$$
\begin{eqnarray}
(S_i \rightarrow E_i) (k) &\sim& \text{Binomial}\Bigg(S_i(k), 1 - \text{exp}\Bigg[- l \beta s_i \sum_{j=1}^{N} N_{c,ij}  \Big( \frac{I_j+A_j}{T_j} \Big) \Bigg]\Bigg)\\
(E_i \rightarrow I_i) (k) &\sim& \text{Binomial}\Bigg(E_i(k), 1 - \text{exp}\Bigg[- l\ \frac{1}{\sigma}\Bigg]\Bigg)\\
(I_i \rightarrow A_i) (k) &\sim& \text{Binomial}\Bigg(I_i(k), 1 - \text{exp}\Bigg[- l\ \frac{a_i}{\omega}\Bigg]\Bigg)\\
(I_i \rightarrow M_i) (k) &\sim& \text{Binomial}\Bigg(I_i(k), 1 - \text{exp}\Bigg[- l\ \frac{1-a_i}{\omega}\Bigg]\Bigg)\\
(A_i \rightarrow R_i) (k) &\sim& \text{Binomial}\Bigg(A_i(k), 1 - \text{exp}\Bigg[- l\ \frac{1}{d_a}\Bigg]\Bigg)\\
(M_i \rightarrow R_i) (k) &\sim& \text{Binomial}\Bigg(M_i(k), 1 - \text{exp}\Bigg[- l\ \frac{1-h_i}{d_m}\Bigg]\Bigg)\\
(M_i \rightarrow ER_i) (k) &\sim& \text{Binomial}\Bigg(M_i(k), 1 - \text{exp}\Bigg[- l\ \frac{h_i}{d_{\text{hospital}}}\Bigg]\Bigg)\\
(ER_i \rightarrow C_i) (k) &\sim& \text{Binomial}\Bigg(ER_i(k), 1 - \text{exp}\Bigg[- l\ \frac{c_i}{d_{\text{ER}}}\Bigg]\Bigg)\\
(ER_i \rightarrow ICU_i) (k) &\sim& \text{Binomial}\Bigg(ER_i(k), 1 - \text{exp}\Bigg[- l\ \frac{1-c_i}{d_{\text{ER}}}\Bigg]\Bigg)\\
(C_i \rightarrow R_i) (k) &\sim& \text{Binomial}\Bigg(C_i(k), 1 - \text{exp}\Bigg[- l\ \frac{1-m_{C,i}}{d_{C,R}}\Bigg]\Bigg)\\
(ICU_i \rightarrow C_{\text{ICU,rec,i}}) (k) &\sim& \text{Binomial}\Bigg(ICU_i(k), 1 - \text{exp}\Bigg[- l\ \frac{1-m_{\text{ICU,i}}}{d_{ICU,R}}\Bigg]\Bigg)\\
(C_{\text{ICU,rec,i}} \rightarrow R_i) (k) &\sim& \text{Binomial}\Bigg(C_{\text{ICU,rec,i}}(k), 1 - \text{exp}\Bigg[- l\ \frac{1}{d_{\text{ICU,rec}}}\Bigg]\Bigg)\\
(C_i \rightarrow D_i) (k) &\sim& \text{Binomial}\Bigg(C_i(k), 1 - \text{exp}\Bigg[- l\ \frac{m_{C,i}}{d_{C,D}}\Bigg]\Bigg)\\
(ICU_i \rightarrow D_i) (k) &\sim& \text{Binomial}\Bigg(ICU_i(k), 1 - \text{exp}\Bigg[- l\ \frac{m_{\text{ICU,i}}}{d_{\text{ICU,D}}}\Bigg]\Bigg)\\
(R_i \rightarrow S_i) (k) &\sim& \text{Binomial}\Bigg(R_i(k), 1 - \text{exp}\Bigg[- l\ \zeta \Bigg]\Bigg)\\
\end{eqnarray}
$$

And the system of equations becomes,

$$
\begin{eqnarray}
S_i(k+1) &=& S_i(k) + (R_i \rightarrow S_i) (k) - (S_i \rightarrow E_i) (k) \\
E_i(k+1) &=& E_i(k) + (S_i \rightarrow E_i) (k) - (E_i \rightarrow I_i) (k) \\
I_i(k+1) &=& I_i(k) + (E_i \rightarrow I_i) (k) - (I_i \rightarrow A_i) - (I_i \rightarrow M_i) (k) \\
A_i(k+1) &=& A_i(k) + (I_i \rightarrow A_i) (k) - (A_i \rightarrow R_i) (k) \\
M_i(k+1) &=& M_i(k) + (I_i \rightarrow M_i) (k) - (M_i \rightarrow R_i) (k) - (M_i \rightarrow ER_i) (k) \\
ER_i(k+1) &=& ER_i(k) + (M_i \rightarrow ER_i) (k) - (ER_i \rightarrow C_i) (k) - (ER_i \rightarrow ICU_i) (k) \\
C_i(k+1) &=& C_i(k) + (ER_i \rightarrow C_i) (k) - (C_i \rightarrow R_i) (k) - (C_i \rightarrow D_i) (k) \\
C_{\text{ICU,rec,i}}(k+1) &=& C_{\text{ICU,rec,i}}(k)  + (ICU_i \rightarrow C_{\text{ICU,rec,i}}) (k) - (C_{\text{ICU,rec,i}} \rightarrow R_i) (k) \\
R_i(k+1) &=& R_i(k) + (A_i \rightarrow R_i) (k)  + (M_i \rightarrow R_i) (k) + (C_i \rightarrow R_i) (k)\\
&& + (C_{\text{ICU,rec,i}} \rightarrow R_i) (k)  - (R_i \rightarrow S_i) (k) \\
D_i(k+1) &=& D_i(k) + (ICU_i \rightarrow D_i) (k) + (C_i \rightarrow D_i) (k) \\
\end{eqnarray}
$$

These equations are implemented in the function `COVID19_SEIRD_sto` located in `src/covid19model/models.py`. The computation itself is performed in the function `solve_discrete` located in `src/covid19model/base.py`. Please note that the deterministic model uses **differentials** in the model defenition and must be integrated, while the stochastic model uses **differences** and must be iterated. The discrete timestep is fixed at one day. The stochastic implementation only uses integer individuals, which is considered an advantage over the deterministic implementation.

### Spatial extension

Rather than considering the entire nation as a homogeneous collection of compartments, it is more realistic to consider spatially distinct regions that are mutually connected. The relevance of this approach becomes clear by looking at the spatial spread in the data (embedded video below): the occurence of confirmed cases is not entirely homogeneous.

<figure class="video_container">
  <video loop="loop" controls="true" poster="_static/figs/First-wave-data_spatial_animated_first.png">
    <source src="_static/figs/First-wave-data_spatial_animated.mp4" type="video/mp4", width="600">
  </video>
</figure>

The main motivation for investigating spatial relations is to quantify the effect of certain types of mobility on the severity and speed of the spread of COVID19, and the effect of spatial isolation of key regions. We consider e.g. the 43 Belgian arrondissements or the 10 Belgian provinces (plus Brussels-Capital). These regions are considered internally homogeneous and are connected through known or estimated mobility matrices.

Below, the spatially explicit model is worked out in a deterministic framework and a stochastic framework. These are largely based on the approach taken in [this 2020 paper](https://www.medrxiv.org/content/10.1101/2020.03.21.20040022v1) by Arenas et al.

#### Deterministic spatial framework

The deterministic approach is largely the same as in the non-spatial framework, only this time every 'patch' (either arrondissement or province) exhibits the dynamics explained before. We make the following assumptions from the mobility:
1. The effective population of a patch is not the same as the registered population, because (during the daytime) some people will leave their home patch and go to another, e.g. for their profession
2. People can get become exposed in their home patch or in the patch they visit
3. The overall mobility is scaled with a single parameter $p$, ranging between 0 and 1, where 0 means no mobility is possible, and 1 indicates regular mobility. This mobility parameter is typically below 1 in a pandemic situation, due to travelling restrictions.
4. Social contact rates are (slightly) different in every patch. Following Arenas et al., this translates to a rescaling of the overall contact rate with respect to density.

Mathematically, the **effective population** in a particular patch $g$ is

$$
\begin{equation}
T^g_{i,\text{eff}} = \sum_{h=1}^G [(1-p_i)\delta^{gh} + p_i P_i^{hg}]T_i^h,
\end{equation}
$$

where $G$ is the total number of patches (e.g. 43 arrondissements), $p_i$ the mobility parameter (stratified per age $i$), $\delta^{gh}$ the Kronecker delta, $P_i^{hg}$ the recurrent mobility *from* patch $h$ *to* patch $g$ (stratified per age $i$), and $T_i^h$ the registered population in age class $i$ and patch $h$. The same formula works goes for $I^g_{i,\text{eff}}$ and $E^g_{i,\text{eff}}$. Note that $T^g_{i,\text{eff}} = T^g_i$ if $p_i = 0$. Note: $P_i^{hg}$ is implemented in the code as `place`.

Because people can get exposed in their own *or* another patch, we need an expression for the **number of susceptibles** from patch $g$ that reside in patch $h$, in age class $i$:

$$
\begin{equation}
    S_i^{gh} = p_i P_i^{gh} S_i^g + (1-p_i) \delta^{gh} S_i^g
\end{equation}
$$

We now have a **social contact matrix** for every patch, $N^g_{c,ij}$. We follow the suggestion in Arenas et al. that the contact rate is proportional to a population density function $f$, or

$$
\begin{equation}
N^g_{c,ij} \propto f\left( \frac{T^g_\text{eff}}{a^g} \right).
\end{equation}
$$

Here $T^g_\text{eff}$ is again the effective population of patch $g$, and $a^g$ its area. The function $f$ is defined as

$$
\begin{equation}
    f(x) = 1 + (1-e^{-\xi x}),
\end{equation}
$$

and increases monotonically from 1 to 2 (see figure below)
<p align="center">
<img src="_static/figs/density-influence.jpg" alt="drawing" width="600"/>
    
This allows us to rewrite $N^g_{c,ij}$:

$$
\begin{equation}
    N^g_{c,ij} = z_i f\left( \frac{T^g_\text{eff}}{a^g} \right) N_{c,ij},
\end{equation}
$$

where Arenas et al. call $z_i$ the 'normalisation factor':

$$
\begin{equation}
    z_i = \frac{ T_i }{ \sum\limits_{h=1}^G f\left(\dfrac{T^h_\text{eff}}{a^h}\right) T^h_{i,\text{eff}}},
    \label{eq:z_i-normalisation}
\end{equation}
$$

with $T_i$ the total population in age class $i$.


Having that background, we define the quantity

$$
\begin{equation}
    B^g_i \equiv \sum_{j=1}^N \beta s_i z_i f\left(\frac{T_\text{eff}^g}{a^g}\right) N_{c,ij} \frac{I_{j,\text{eff}}^g + A_{j,\text{eff}}^g}{T_{j,\text{eff}}^g}
\end{equation}
$$

The **coupled ordinary differential equations** can be rewritten to the spatial situation

$$
\begin{eqnarray}
\dot{S}_i^g &=& - \sum_{h=1}^G S_i^{gh} B^h_i + \zeta R_i^g, \\
\dot{E}_i^g &=& \sum_{h=1}^G S_i^{gh} B^h_i - (1/\sigma) \cdot E_i^g,  \\
\dot{I}_i^g &=& (1/\sigma) E_i - (1/\omega) I_i^g, \\
\dot{A}_i^g &=& (\text{a}_i/\omega) I_i^g - (1/d_{\text{a}}) A_i^g, \\
\dot{M}_i^g &=&  ((1-\text{a}_i) / \omega ) I_i^g - ( (1-h_i)/d_m + h_i/d_{\text{hospital}} ) M_i^g, \\
\dot{ER}_i^g &=& (h_i/d_{\text{hospital}}) M_i^g - (1/d_{\text{ER}}) ER_i^g, \\
\dot{C}_i^g &=& c_i (1/d_{\text{ER}}) ER_i^g  - (m_{C, i}/d_{c,D}) C_i^g - ((1 - m_{C, i})/d_{c,R}) C_i^g, \\
\dot{ICU}+i^g &=& (1-c_i) (1/d_{\text{ER}}) ER_i^g - (m_{ICU,i}/d_{\text{ICU},D}) ICU_i^g  \\
&& - ((1-m_{ICU,i})/d_{\text{ICU},R}) ICU_i^g,\\
\dot{C}_{\text{ICU,rec,i}}^g &=& ((1-m_{ICU,i})/d_{\text{ICU},R}) ICU_i^g - (1/d_{\text{ICU,rec}}) C_{\text{ICU,rec,i}}^g, \\
\dot{D}_i^g &=&  (m_{ICU,i}/d_{\text{ICU},D}) ICU_i^g +  (m_{C,i}/d_{\text{c},D}) C_i^g , \\
\dot{R}_i^g &=&  (1/d_a) A_i + ((1-h_i)/d_m) M_i^g + ((1-m_{C,i})/d_{c,R}) C_i^g \\
&& + (1/d_{\text{ICU,rec}}) C_{\text{ICU,rec,i}}^g - \zeta R_i^g,
\end{eqnarray}
$$

All state matrices have dimensions $G \times N$, so integrating these ODEs takes (more than) $G$ times the computation time compared to the non-spatial case. Note that, if relevant and available, all parameters may be further stratified per patch $g$.

These equations are implemented in the function `COVID19_SEIRD_spatial` located in `src/covid19model/models.py`. When creating the simulation, these equations are integrated numerically.

#### Stochastic spatial framework

Extending the deterministic spatial framework to a (discrete) stochastic one is straightforward. The probability of being infected in patch $g$ in the next timestep (stepsize $\ell$) for a susceptible individual is

$$
\begin{equation}
    P^g_i = 1 - \exp\left( - \ell \sum\limits_{j=1}^N \beta s_i z_i f\left( \frac{T_\text{eff}^g}{a^g} \right) N_{c,ij} \frac{ I^{g}_{j, \text{eff}} + A^{g}_{j, \text{eff}}}{T_{j,\text{eff}}^g} \right)
\end{equation}
$$

The same delocalised infection happens here, again depending on the overall mobility $p_i$ and the recurrent mobility matrix $P_i^{gh}$:

$$
\begin{equation}
    \Pi_i^g = P_i^g + p_i\sum\limits_{\substack{h=1 \\ h\neq g}}^G P_i^{gh}P_i^h
\end{equation}
$$

The **overall probability of exposure** for an agent in age class $i$ and patch $g$, $\Pi_i^g$, is a sum of
1. The probability of being infected at home, *plus*
2. The probability of being infected outside the home patch, weighed by the mobility to that patch

Note that a higher mobility parameter does *not* necessarily mean that the exposure probability $\Pi_i^g$ is larger, because the probability of becoming infected in the home patch, $P_i^g$, is also dependant on the mobility parameter $p_i$.

This translates to the same **set of stochastic equations** as in the non-spatial case, but now with spatial stratification and a slightly adjusted probability in the binomial experiment:

$$
\begin{eqnarray}
(S_i^g \rightarrow E_i^g) (k) &\sim& \text{Binomial}\Bigg(S_i^g(k), \Pi_i^g(k) \Bigg)\\
(E_i^g \rightarrow I_i^g) (k) &\sim& \text{Binomial}\Bigg(E_i^g(k), \Pi_i^g(k) \Bigg)\\
\end{eqnarray}
$$


These in turn determine the discrete development of the stochastic model:

$$
\begin{eqnarray}
S_i^g(k+1) &=& S_i^g(k) + (R_i^g \rightarrow S_i^g) (k) - (S_i^g \rightarrow E_i^g) (k) \\
E_i^g(k+1) &=& E_i^g(k) + (S_i^g \rightarrow E_i^g) (k) - (E_i^g \rightarrow I_i^g) (k) \\
\end{eqnarray}
$$

... etcetera (see non-spatial case).

An **example** of the stochastic spatial framework is shown in the video below.

<figure class="video_container">
  <video loop="loop" controls="true" poster="_static/figs/demo_arr_demo_arr_5-35-yo_81000.png">
    <source src="_static/figs/demo_arr_demo_arr_5-35-yo_81000.mp4" type="video/mp4" width="600">
  </video>
</figure>
<em>Simulation with the stochastic model and an initial condition where at time = 0, 5 exposed individuals in age class 30-40 years are 'released' in Arlon, all the way on the south-easter side of the country. At the onset, it is clear how the 'wave' of new exposures travels through the country, hitting the densely populated patches first. At day 40, measures are taken that bring down the effective reproduction constant below zero.</em>

These equations are implemented in the function `COVID19_SEIRD_sto_spatial` located in `src/covid19model/models.py`. The computation itself is performed in the function `solve_discrete` located in `src/covid19model/base.py`. Please note that the deterministic model uses **differentials** in the model definition and must be integrated, while the stochastic model uses **differences** and must be iterated. The discrete timestep is fixed at one day. The stochastic implementation only uses integer individuals, which is considered an advantage over the deterministic implementation.

#### Spatial: remaining issues and tasks

The theory at the basis of both the deterministic and the stochastic spatial model is still subject to change. Some of these 'to-dos' are listed below.

1. In the original implementation of the spatial method, the type of contact in the exposure probability was related to whether or not the contact takes place in the home patch or not (see equation below, in the original notation). The reason is that e.g. home contact ($N_{\text{home},ij}$ is not relevant when visiting other places; if the individual does not work in his home patch, the work vs home, school, leisure, other human-to-human contacts must be seperate. Whilst this is certainly true, it is currently unclear whether this additional complication is worth pursuing.

$$
\begin{equation*}
    P(S_{i,g} \rightarrow E_{i,g}) (k) = 1 - \text{exp} \Bigg[ \underbrace{\text{P}_{g,g} \Bigg\{ - l \beta s_i \sum_{j=1}^{N} N_{\text{c, tot, ij}} \Bigg( \frac{I_{j,g} + A_{j,g}}{T_{j,g}} \Bigg) \Bigg\}}_{\text{individual working in residence patch}}  \\
    \quad + \sum_{l=1\\ l \neq g}^{G} \text{P}_{g,l} \Bigg\{ \underbrace{- l \beta s_i \sum_{j=1}^{N} N_{\text{c, work, ij}} \Bigg( \frac{I_{j,l} + A_{j,l}}{T_{j,l}} \Bigg)}_{\text{work interactions in work patch (subscript l)}} \\
     \quad - \underbrace{l \beta s_i \sum_{j=1}^{N} (N_{\text{c, home, ij}} + N_{\text{c, school, ij}} + N_{\text{c, leisure, ij}} + N_{\text{c, others, ij}}) \Bigg( \frac{I_{j,g} + A_{j,g}}{T_{j,g}} \Bigg)}_{\text{all other interactions in home patch (subscript g)}} \Bigg\} \Bigg]
\end{equation*}
$$

2. Many parameters, most notably the mobility parameter $p_i$, can (and often should) be stratified further.
3. The calibration methods are still under construction and are more difficult, mainly because of the higher number of possible choices:
    * Should every patch have its own $\beta^g$ value?
    * How do we go about choosing an initial condition for the calibration?
4. A good spatial model is largely dependant on good mobility data. The currently available data is a mobility matrix from 2011, which is quite certainly not representative for the 2020 pandemic situation. Ideally we have *current* mobility data, stratified per *age class* and per *activity* (work/school/leisure/...) 
5. This model currently does *not* account for importation of exposed individuals, which in a highly populated region like Western Europe is undoubtedly an important factor.
6. The contact rate's dependence on the density may be omitted -- it has little effect -- but is currently implemented to parallel Arenas et al.'s approach.
7. Arguably, stratification at the level of arrondissements is too highly detailed in a 'big city' like Belgium. Data analysis of the current situation may point out that it is justified to aggregate some arrondissements with a similar time series morphology together.

### Transmission rates and social contact data

In our model, the transmission rate of the disease depends on the product of four contributions. The first contribution, $(I+A)/T$, is the fraction of contagious individuals in the population. The second contribution, $\mathbf{N}_c$, is the average number of human-to-human interactions per day. The third contribution, $s_i$, is the relative susceptiblity to SARS-CoV-2 infection in age group $i$, and the fourth contribution, $\beta$, is the probability of contracting COVID-19 when encountering a contagious individual under the assumption of 100 \% susceptibility to SARS-CoV-2 infection. We assume that the per contact transmission probability $\beta$ is independent of age and we will infer its distribution by calibrating the model to national Belgian hospitalization data. The number of human-human interactions, $\mathbf{N}_c$, are both place and age-dependent. These matrices assume the form of a 9x9 *interaction matrix* where an entry X, Y denotes the number of social contacts age group X has with age group Y per day. These matrices are available for homes, schools, workplaces, in public transport, and leisure activities, from a survey study by Lander Willem (2012). The total number of social interactions is given by the sum of the contributions in different places,

$$
\begin{equation}\label{eq:interaction_matrices}
\mathbf{N_{\text{c}}} = \mathbf{N_{\text{c, home}}} + \mathbf{N_{\text{c, schools}}} + \mathbf{N_{\text{c, work}}} + \mathbf{N_{\text{c, transport}}} + \mathbf{N_{\text{c, leisure}}} + \mathbf{N_{\text{c, others}}}.
\end{equation}
$$

Coefficients can be added to the contributing contact matrices to model a goverment policy. For instance, to model the Belgian lockdown, the mobility reductions deduced from the Google community mobility reports were used as coefficients for the different interaction matrices. We assumed workplace interactions were down to only 40 % of their prepandemic values before the lockdown.

### Modeling social intertia

The model takes into account the effect of *social inertia* when measures are taken. In reality, social restrictions or relaxations represent a change in behaviour which is gradual and cannot be modeled using a step-wise change of the social interaction matrix $\mathbf{N_c}$. This can be seen when closely inspecting the *Google community mobility report* above. Multiple functions can be used to model the effects of social compliance, e.g. a delayed or non-delayed ramp, or a logistic function. In our model, we use a delayed ramp to model compliance,

$$
\begin{equation}
\mathbf{N_{c}}^{k} = \mathbf{N_{\text{c, old}}} + f^{k} (\mathbf{N_{\text{c, new}}} - \mathbf{N_{\text{c, old}}})
\end{equation}
$$

where,

$$
\begin{equation}
    f^k=
\begin{cases}
	0.0,& \text{if } k\leq \tau\\
    \frac{k}{l} - \frac{\tau}{l},& \text{if } \tau < k\leq \tau + l\\
    1.0,              & \text{otherwise}
\end{cases}
\end{equation}
$$

where $\tau$ is the number of days before measures start having an effect and $l$ is the number of additional days after the time delay until full compliance is reached. Both parameters were calibrated to the daily number of hospitalizations in Belgium (notebooks `notebooks/0.1-twallema-calibration-deterministic.ipynb` and `notebooks/0.1-twallema-calibration-stochastic.ipynb`). $k$ denotes the number of days since a change in social policy.

### Basic reproduction number

The basic reproduction number $R_0$, defined as the expected number of secondary cases directly generated by one case in a population where all individuals are susceptible to infection, is computed using the next generation matrix (NGM) approach introducted by Diekmann. For our model, the basic reproduction number of age group $i$ is,

$$
\begin{equation}\label{eq:reproduction_number}
R_{0,i} = (a_i d_a + \omega) \beta s_i \sum_{j=1}^{N} N_{c,ij}
\end{equation}
$$

and the population basic reproduction number is calculated as the weighted average over all age groups using the demographics of Belgium. The detailed algebra underlying the computation equation of the basic reproduction number is presented in the supplementary materials of our manuscript (see section *Previous work*).

### Model parameters

An in-depth motivation of the model parameters is provided in our manuscript (see section *Previous work*). Hospital parameters were derived from a dataset obtained from two hospitals in Ghent (Belgium).

<p align="center">
<img src="_static/figs/parameters.png" alt="parameters" width="1200"/>
<em> Overview of BIOMATH COVID-19 model parameters. </em>
</p>
