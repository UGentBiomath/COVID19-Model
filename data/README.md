## Data sets for COVID 19 model

### Raw data sets

#### Economic data

+ `input-output.xlsx` contains sectoral input-ouput tables for Belgium. Belgian data, NACE 64 classification. Received from prof. Gert Peersman.
+ `Employees_25-04-2020_NACE38.xlsx` contains the fraction of employees who worked at the workplace, at home, in a mix of both, those temporarely unemployed and those abscent during the Belgian lockdown of March 17th, 2020 to March 4th, 2020. Belgian data, NACE 38 classification. Received from prof. Gert Peersman. Survey performed by the Belgian national bank.
+ `Employees_NACE38.xlsx` contains the number of employees per sector from 2014 to 2018. Belgian data, NACE 38 classification. Retrieved from http://stat.nbb.be/?lang=nl, 'Bevolking en arbeidsmarkt' > 'Werkgelegenheid' > 'Binnenlands concept A38'.
+ `Employees_NACE64.xlsx` contains the number of employees per sector from 2014 to 2018. Belgian data, NACE 38 classification. Retrieved from http://stat.nbb.be/?lang=nl, 'Bevolking en arbeidsmarkt' > 'Werkgelegenheid' > 'Binnenlands concept A64'.
+ `table_ratio_inv_go.csv` contains, for every sector in the WIOD 55 classification, the number of days production can continue when no inputs are delivered (= stock). Retrieved from https://zenodo.figshare.com/articles/software/Production_networks_and_epidemic_spreading_How_to_restart_the_UK_economy_/12345527/1
+ `WIOD_shockdata.csv` contains estimated household and other demand shocks during an economic crisis. Retrieved from https://zenodo.figshare.com/articles/software/Production_networks_and_epidemic_spreading_How_to_restart_the_UK_economy_/12345527/1
+ `IHS_Markit_results_compact.csv` Criticality scores of IHS Markit analysts. The exact formulation of the question was as follows: “For each industry in WIOD 55, please rate whether each of its inputs are essential. We will present you with an industry X and ask you to rate each input Y. The key question is: Can production continue in industry X if input Y is not available for two months?” UK data, WIOD 55 classification. Retrieved from https://zenodo.figshare.com/articles/software/Production_networks_and_epidemic_spreading_How_to_restart_the_UK_economy_/12345527/1

#### GIS

+ `NIS_name.csv` is a two way NIS-name table used for the function name2nis located in `src/models/utils.py`. It is a pandas dataframe with two columns: the first are the Belgian NIS codes, the second is the name corresponding to that NIS code.

+ `NIS_arrondissement.csv` : NIS-code of each arrondissement

+ `NIS_Province.csv` : NIS-code of each province

+ `arrondissements_per_province.csv` : province to which each arrondissement belongs

+ `inhabitants.csv` : number of inhabitants for each municipality, arrondissement and region  

+ `shapefiles/BE/...` :  shapefiles of Belgian municipalities, district, provinces, regions, arronddissements

#### Hospital data

+ `symptomOnsetHospitalization.xlsx` contains: 1) the date at which patients first reported having symptoms, 2) the data at which patients were hospitalized and 3) the age of the patient. Received from Ghent University hospital, contact: pascal.coorevits@uzgent.be .

+ `AZmariaMiddelares.xlsx` contains: 1) patient ID, 2) age and sex of patient, 3) per patient: in chronological order, from bottom to top (!), the amount of time spent in the emergency room, cohort or intensive care unit, 4) if the patient recovered or died. 'cohortafdeling D601' is a geriatric cohort ward, 'cohortafdeling D501' is a regular cohort ward. Received from AZ Maria Middelares, contact: Leen Van Hoeymissen (Leen.VanHoeymissen@AZMMSJ.BE).

+ `UZGent_full.xlsx` contains: 1) patient ID, 2) age and sex of patient, 3) per patient: the date of symptom onset, date of first assessment, date of first hospital contact, the admission date to the Ghent University hospital, the admission data to ICU, the discharge date in ICU, the discharge date from the Ghent University hospital. 5) if the patient recovered or died. Dataset received 05/07/2020 from the Ghent University Hospital, contact: prof. Ernst Rietzschel (ernst.rietzschel@ugent.be).

#### Interaction matrices

##### Willem 2012

+ `total.xlsx`, `home.xlsx`, `work.xlsx`, `leisure.xlsx`, `transport.xlsx`, `school.xlsx`, `otherplace.xlsx`:  contains the interaction matrix (in the place suggested by the spreadsheets name) based on a survey study in Flanders with 1752 participants. The spreadsheet has several tabs to distinguish between the nature and duration of the contact. The data were extracted using the social contact rates data tool made by Lander Willem, available at https://lwillem.shinyapps.io/socrates_rshiny/. For the extraction of the data, weighing by age, weighing by week/weekend were used and reciprocity was assumed. Contacts with non-household members are defined as leisure contacts instead of home contacts. 

##### CoMiX

+ `wave1.xlsx`, ..., `wave8.xlsx` : contain the interaction matrices under lockdown measures in Belgium. There is one spreadsheet per survey wave. The dates of the surveys were (wave 1 - 8): ['24-04-2020','08-05-2020','21-05-2020','04-06-2020','18-06-2020','02-07-2020','02-08-2020','02-09-2020'].  Each spreadsheet has two tabs to distinguish between the nature and duration of the contact. The data were extracted using a beta version of the social contact rates data tool made by Lander Willem, SOCRATES. The data are not yet publically available. For the extraction of the data, weighing by age, weighing by week/weekend were used and reciprocity was assumed.

##### Demographic

+ `Age pyramid of Belgium.csv` contains the most recent (January 1st, 2020) age pyramid of Belgium. Given as the number of individuals per gender and per 5 year age bins. Retreived from https://statbel.fgov.be/en/themes/population/structure-population

+ `TF_SOC_POP_STRUCT_2020.xlsx` contains the most recent (January 1st, 2020) structure of the Belgian population, per age and per municipality. This is used to extract the spatially and age-stratified population `initN`. Retrieved from https://statbel.fgov.be/nl/open-data/bevolking-naar-woonplaats-nationaliteit-burgerlijke-staat-leeftijd-en-geslacht-10

#### Google

+ `community_mobility_data_BE.csv` contains a copy of the Google Community Mobility Report (Belgium only) dataset downloaded by the function `get_google_mobility_data()`. Mobility data is extracted from https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=2dcf78defb92930a

#### Sciensano

+ `COVID19BE_HOSP.csv` contains a copy of the "HOSP" sheet from the publically available Sciensano data. Data is extracted from https://epistat.sciensano.be/Data/COVID19BE.xlsx
+ `ObsInf.txt` contains cumulative observed infections from 1 March on
 (note that this is an underestimation since especially in the beginning, only sick people
   were tested)
+ `ObsDeceased.txt` contains cumulative observed deaths from 1 March on
+ `ObsRecovered.txt` contains cumulative observed recovered (from hospital) from 1 March on

#### Model parameters

+ `verity_etal.csv` contains age-stratified estimates of the number of symptomatic cases which result in hospitalization, the number of hospitalized patients in need of intensive care and the case fatality ratio (the percentage of individuals with symptomatic or confirmed disease who die from the disease). Data were obtained from https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext
+ `davies_etal.csv` contains age-stratified estimates of the fraction of asymptomatic cases and relative susceptibility. Data were copied from https://www.nature.com/articles/s41591-020-0962-9
+ `others.csv` contains all other parameters used to run the model. Obtained from various sources, detailed in the report.
+ `molenberghs_etal.csv` contains age-stratified probability to die after infection,
split between general population and elderly homes. Data from https://m.standaard.be/cnt/dmf20200609_04985767. Paper: https://www.medrxiv.org/content/10.1101/2020.06.20.20136234v1.
+ `non_stratified.csv` contains non-stratified estimates of asymptomatic cases, symptomatic cases which result in hospitalization, hospitalized patients in need of intensive care and the case fatality ratio

#### Belgian Census 2011

+ `Pop_LPW_NL_25FEB15.xlsx` contains the working population per sex, place of residence and place of work. First, the raw spreadsheet `data/raw/census_2011/Pop_LPW_NL_25FEB15.xlsx` was modified in MS Excel and placed in the data/interim folder under the name `Pop_LPW_NL_25FEB15_delete_unknown.xlsx`. Data free for download at https://census2011.fgov.be/download/downloads_nl.html .
+ `census_demo_nl_04nov14.xlsx` contains all demographic data from the 2011 Belgian census. From these data, the number of individuals in 10 year age-bins per Belgian arrondissement are calculated. The conversion notebook is `notebooks/0.1-twallema-extract-census-data.ipynb`.  Data free for download at https://census2011.fgov.be/download/downloads_nl.html .
+ `census_arbeidsmarkt_nl_24oct14.xlsx` contains all work related data from the 2011 Belgian census. Data free for download at https://census2011.fgov.be/download/downloads_nl.html .

#### QALY model

+ `Life_tables_Belgium.csv` contains belgiam life tables for different years and each gender. A copy containing only the necessary information for the most recent year (2019) was placed in the data/interim folder under the name: 'Life_table_Belgium_2019.csv'.  Data obtained from: https://statbel.fgov.be/nl/themas/bevolking/sterfte-en-levensverwachting/sterftetafels-en-levensverwachting .
+ `QoL_scores_Belgium_2018.csv` contains quality of life scores for the Belgian population calculated from the 2018 health survey under the EuroQOL 5 scale. The data was interpoletad to fit the main model's age bins and was placed in the data/interim under the name: 'QoL_scores_Belgium_2018_v1.cs'. Data obtained from: https://hisia.wiv-isp.be/SitePages/Home.aspx .
+ `costs_hospital_belgium.csv` contains the reported total costs of medical treatment per disease category in Belgium for 2018. The data was corrected for inflation, combined with cost per QALY information and placed in data/interim folder under the name: 'hospital_data_qaly.xlsx'.  Data obtained from: https://tct.fgov.be/webetct/etct-web/html/fr/index.jsp .
+ `hec03946-sup-0001-supplementary material.docx` contains supply-side cost-effectiveness thresholds and elasticities per disease group and age for the Netherlands. The data was used to estimate the cost per QALY gained per disease group. It was subsequently corrected for inflation, combined with costs of medical treatment and  and placed in data/interim folder under the name: 'hospital_data_qaly.xlsx'. Suplementary material of :Stadhouders, N., Koolman, X., Dijk, C., Jeurissen, P., and Adang, E. (2019). The marginal benefits of healthcare spending in the Netherlands: Estimating cost-effectiveness thresholds using a translog production function. Health Economics, 28(11):1331–1344.

### Interim data sets conversion scripts

Conversion scripts are managed inside the `covid19model` package (`src/covid19model/data` folder).

#### Interaction matrices

##### Willem 2012

+ `total.xlsx`, `home.xlsx`, `work.xlsx`, `leisure.xlsx`, `transport.xlsx`, `school.xlsx`, `otherplace.xlsx`:  contains the interaction matrix (in the place suggested by the spreadsheets name) based on a survey study in Flanders with 1752 participants. Two sheets were added to every corresponding spreadsheet in the raw folder: 1) interactions longer than 5 minutes and 2) interactions longer than 15 minutes. These were computed as the total number of interactions minus the interaction lasting less than 5/15 minutes.

##### Demographic

+ `BELagedist_5year.txt` contains the most recent (January 1st, 2020) age pyramid of Belgium. Given as the number of individuals per 5 year age bins irrespective of gender. Converted from raw dataset `Age pyramid of Belgium.csv` using MS Excel.

+ `BELagedist_10year.txt` contains the most recent (January 1st, 2020) age pyramid of Belgium. Given as the number of individuals per 10 year age bins irrespective of gender. Converted from raw dataset `Age pyramid of Belgium.csv` using MS Excel.

#### Hospital data
+ `twallema_AZMM_UZG.xlsx` contains the merged dataset from AZ Maria Middelares and Ghent University hospital. The combined samplesize is 370 patients. The resulting dataset contains the following entries: 1) age of patient, 2) sex of patient, 3) type of stay. Emergency room only, Cohort only or ICU. Here, ICU implies that the patient spent a limited time in Cohort before transitioning to an ICU unit and if not deceased in ICU, the patient returns to Cohort for recovery, 4) outcome (R: recovered, D: deceased), 5) dC: time spent in a Cohort ward, 6) dICU: time spent in an ICU, 7) dICUrec: time spent in Cohort recovering after an ICU stay. Code of reformat performed in `notebooks/0.1-twallema-AZMM-UZG-data-analysis.ipynb`.

#### Model parameters

+ `AZMM_UZG_hospital_parameters.csv` contains age-stratified estimates for the following model parameters: 1) c: probability of not going to an ICU where (1-c) is the probability of needing IC. 2) m0: mortality, given as a total (cohort + ICU) and separate for Cohort and ICU. 3) dC: average time spent in a Cohort ward if not going to ICU. Split in recovered and deceased. 4) dICU: average time spent in an ICU. Split in recovered and deceased. 4) dICU,rec: average length of recovery stay in Cohort after ICU. Code of reformat performed in `notebooks/0.1-twallema-AZMM-UZG-data-analysis.ipynb`.

+ `deterministic_22072020.json` contains the posterior distributions of the calibrated model parameters. The distributions are associated with the following preprint: `https://doi.org/10.1101/2020.07.17.20156034`.

#### Belgian Census 2011
+ `Pop_LPW_NL_25FEB15_delete_unknown.xlsx`. First, the raw spreadsheet `data/raw/census_2011/Pop_LPW_NL_25FEB15.xlsx` was modified in MS Excel. The format of the data is as follows:
    - rows: municipality of residence
    - columns: municipality of work   
The dataset contained, for each Belgian province, a column of 'unknowns', indicating we know where these individuals live but not where they work. These 10 columns were removed manually. Further, the column `Werkt in Belgie` was renamed `Belgie` to make name-based row and column matching easier. The recurrent mobility matrix was extracted from these data. The conversion notebook is `notebooks/0.1-twallema-extract-census-data.ipynb`.

+ `recurrent_mobility.csv` contains a square recurrent mobility matrix of the Belgian arrondissements (43x43). The data were extracted from `Pop_LPW_NL_25FEB15_delete_unknown.xlsx`, the conversion was performed in `notebooks/0.1-twallema-extract-census-data.ipynb`. This data is deprecated since 2019.

+ `census-2011-updated_row-commutes-to-column_arrondissements.csv` contains a square (but non-symmetric) mobility matrix of the Belgian arrondissements (43x43). The data were extracted from `Pop_LPW_NL_25FEB15_delete_unknown.xlsx`.

+ `census-2011-updated_row-commutes-to-column_municipalities.csv` contains a square (but non-symmetric) mobility matrix of the Belgian municipalities (581x581). The data were extracted from `Pop_LPW_NL_25FEB15_delete_unknown.xlsx`.

+ `census-2011-updated_row-commutes-to-column_provinces.csv` contains a square (but non-symmetric) mobility matrix of the Belgian provinces *and* arrondissement Brussels-Capital (NIS 21000) (11x11). The data were extracted from `Pop_LPW_NL_25FEB15_delete_unknown.xlsx`.

+ `census-20110-updated_row-commutes-to-column_test.csv` contains a square (but non-symmetric) mobility matrix of the three Belgian arrondissements (Antwerpen, Brussel, Gent) (3x3). This is an artificial case: all commuters that leave the home arrondissement but do *not* go to one of the other two arrondissements, have been counted as staying at the home arrondissement instead. The data were extracted from `Pop_LPW_NL_25FEB15_delete_unknown.xlsx`.

#### QALY model

+ `Life_table_Belgium_2019.csv` contains the probability of dying at a given age for the Belgian population as of 2019.  
+ `QoL_scores_Belgium_2018_v3.csv` contains age-stratified quality of life scores for the Belgian population calculated from the 2018 health survey under the EuroQOL 5 scale.
+ `hospital_data_qaly.xlsx` contains the total reported costs of hospital healthcare in Belgium per disease group as well as the estimated cost per QALY gained for the same groups.

#### Demographic data

+ `age_structure_per_arrondissement.csv` : population of each age per arrondissement

+ `age_structure_per_municipality.csv` : population of each age per municipality

+ `age_structure_per_province.csv` : population of each age per province

+ `age_structure_per_test.csv` : population for the test case: only arrondissements Antwerp, Brussels, Gent

+ `area_arrond.csv` contains the area of Belgian arrondissements per NIS code in square meters

+ `area_municip.csv` contains the area of Belgian municipalities per NIS code in square meters

+ `area_province.csv` contains the area of Belgian provinces per NIS code in square meters

+ `area_test.csv` contains the area of arrondissements Antwerp, Brussels, Gent

+ `initN_arrond.csv` contains a CSV with the following columns: arrondissement NIS-code, total population, population aged 0-9, population aged 10-19, ..., population aged 80 and above. Created in `notebooks/JV-extract-age-structures.ipynb`.

+ `initN_province.csv` contains a CSV with the following columns: province NIS-code, total population, population aged 0-9, population aged 10-19, ..., population aged 80 and above. Created in `notebooks/JV-extract-age-structures.ipynb`.

+ `initN_municip.csv` contains a CSV with the following columns: municipality NIS-code, total population, population aged 0-9, population aged 10-19, ..., population aged 80 and above. Created in `notebooks/JV-extract-age-structures.ipynb`.

+ `initN_test.csv` contains a CSV with the following columns: NIS-code of arrondissement Antwerpen, Brussel, Gent, total population, population aged 0-9, population aged 10-19, ..., population aged 80 and above. Created in `notebooks/JV-extract-age-structures.ipynb`.

#### Economic data

All economic data from the data/raw/economical was converted in the notebook `notebooks/twallema-extract-economic-data.ipynb`.

+ `conversion_matrices.xlsx` contains conversion matrices to more easily aggregate data from different sector classifications. F.i. converting from NACE 64 to WIOD 55 classification.
+ `census2011_NACE21.csv` contains per Belgian arrondissement (43 in total) the number of employees in every sector of the NACE 21 classification.
+ `others.csv` contains the sectoral output during business-as-usual, household demand during business-as-usual, other final demand during business-as-usual, the desired stock, consumer demand shock, other demand shock, sectoral employees during business-as-usual and sectoral employees under lockdown. Data from various sources. NACE 64 classification.
+ `IO_NACE64.csv` contains the input-output table for Belgium, formatted to NACE 64 classification.
+ `IHS_critical_NACE64.csv` contains the IHS Market Analysts data, reformatted from WIOD 55 to the NACE 64 classification.


### simulated

Contains zarr directories, which in turn contain groups that each hold a different simulation result. The aim of this 'simulation database' is to be able to save simulation results and perform post-processing without always having to go through the long and computationally demanding task of simulating (using the base.py sim function). This is especially relevant for spatially stratified SEIRD extended models, as these typically take G times longer to run (where G is the level of spatial stratification).

Simulations are saved here using the `utils.py` function `save_sim()` and opened using the `open_sim()` function. The content of the simulations is suggested in the directory titles and mentioned in the 'description' attribute of the zarr groups. It is also printed upon opening the simulation with `open_sim()`. Additionally, the simulation is quickly described here as well.

#### Sanity-check_100sims_100days_Nctot-to-Nchome-day40.zarr

+ `arr_1E-per-arr` description: "Stochastic spatial SEIRD extended model with 100 parallel simulations in 43 arrondissements over 100 days.At day 0 a single exposed person in the age class 30-40 is released ineach of the arrondissements. At day 40 measures are imposed, bringing down the contact rate from Nc_total to Nc_home over the course of 5 + 5 days (tau and l compliance parameters) reaches full effect."