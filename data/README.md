## Data sets for COVID-19 model

### Raw data sets

#### deaths

+ `DEMO_DEATH_OPEN.xlsx` contains total number of daily deaths per age class per arrondissement between 2009 and now. Current latest date: 21 February 2021. Data for 2020 and 2021 are not entirely complete yet. Downloaded from Statbel (https://statbel.fgov.be/nl/open-data/aantal-sterfgevallen-dag-geslacht-arrondissement-leeftijd)

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

+ `Postcode_Niscode` : dictionary that injectively relates postal codes and NIS codes.

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

#### Mobility
##### Google

+ `community_mobility_data_BE.csv` contains a copy of the Google Community Mobility Report dataset downloaded by the function `get_google_mobility_data()`. Mobility data is extracted from https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=2dcf78defb92930a Only data for Belgium is saved in `data/raw`, because the "global" file is over 250 Mb.

##### Apple
+ `apple_mobility_trends.csv` contains a copy of the Apple Mobility Trends downloaded by the function `get_apple_mobility_data()`. Mobility data is extracted from https://covid19-static.cdn-apple.com/covid19-mobility-data/2024HotfixDev12/v3/en-us/applemobilitytrends-2021-01-10.csv

##### Proximus
+ `outputPROXIMUS122747coronaYYYYMMDDAZUREREF001.csv`: complete origin-destination data saved on SFTP server and locally but NOT uploaded to GitHub.

#### Sciensano

+ `COVID19BE_HOSP.csv` contains a copy of the "HOSP" sheet from the publically available Sciensano data. Data is extracted from https://epistat.sciensano.be/Data/COVID19BE.xlsx
+ `COVID19BE_MORT.csv` contains a copy of the "MORT" sheet from the publically available Sciensano data. Data is extracted from https://epistat.sciensano.be/Data/COVID19BE.xlsx
+ `COVID19BE_CASES.csv` contains a copy of the "CASES_AGESEX" sheet from the publically available Sciensano data. Data is extracted from https://epistat.sciensano.be/Data/COVID19BE.xlsx
+ `COVID19BE_VACC.csv` contains a copy of the "VACC" sheet from the publically available Sciensano data. Data is extracted from https://epistat.sciensano.be/Data/COVID19BE.xlsx
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

#### VOCs

+ `sequencing_501YV1_501YV2_501YV3.csv` contains the total number of sequenced samples and the number of samples of variants 501Y.V1 (British), 501Y.V2 (South African) and 501Y.V3 (Brazilian). Data available from week 49 of 2020 until week 14 of 2021. Data download from Tom Wenseleer's Git (https://github.com/tomwenseleers/newcovid_belgium); folder `~/data/2021_04_16/sequencing_501YV1_501YV2_501YV3.csv`.

#### Seroprelevance

+ `Belgium COVID-19 Studies - Sciensano_Blood Donors_Tijdreeks.csv` contains the overall seroprelevance in the Belgian population with confidence bounds. Downloaded from https://datastudio.google.com/embed/reporting/7e11980c-3350-4ee3-8291-3065cc4e90c2/page/R4nqB on 2021-04-19 by right clicking on a datapoint and selecting 'csv downloaden'.
+ `serology_covid19_belgium_round_1_to_7_v20210406.csv` contains the measured amount of antibodies of samples per province, per age group and in seven sampling waves. Downloaded from https://zenodo.org/record/4665373#.YH116nUzaV4 on 2021-04-19.

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

+ `AZMM_UZG_hospital_parameters.csv` contains age-stratified estimates for the following model parameters: 1) c: probability of not going to an ICU where (1-c) is the probability of needing IC. 2) m0: mortality, given as a total (cohort + ICU) and separate for Cohort and ICU. 3) dC: average time spent in a Cohort ward if not going to ICU. Split in recovered and deceased. 4) dICU: average time spent in an ICU. Split in recovered and deceased. 4) dICU,rec: average length of recovery stay in Cohort after ICU. The analysis is performed using the script `~/notebooks/preprocessing/AZMM-UZG-hospital-data-analysis`.
+ `sciensano_hospital_parameters.csv` contains age-stratified estimates for the hospital parameters of the COVID19_SEIRD model. The analysis was performed using the script `~/notebooks/preprocessing/sciensano-hospital-data-analysis`. You must place the super secret detailed hospitalization dataset `COVID19BE_CLINIC.csv`in the same folder as this script in order to run it.
+ `wu_asymptomatic_fraction.xlsx` contains the computation of the symptomatic fraction of the Belgian population per age group based on a study of Wu, 2020 (https://www.nature.com/articles/s41591-020-0822-7/figures/2).


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

#### Sciensano

+ `clusters.csv` contains the number of clusters traced back to 1) families, 2) WZCs, 3) schools, 4) workplaces and 5) others over the period 2020-12-28 until 2021-02-21. Data extracted from the weekly Sciensano reports, available at https://covid-19.sciensano.be/nl/covid-19-epidemiologische-situatie. Last visited 2021-03-31.
+ `sciensano_detailed_mortality.csv` contains the number of deaths (incidence and cumulative number) per age group, in total, in hospitals and in nursing homes. Since our model does not predict nursing home deaths, model output must be compared to deaths in hospitals. Data conversion was done using the script` ~/notebooks/preprocessing/sciensano-mortality-data-analysis.py`. You must place the super secret detailed hospitalization dataset `COVID19BE_MORT_RAW.csv` in the same folder as this script in order to run it. Permission of Sciensano is needed to obtain the raw dataset.


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

All economic data from the data/raw/economical was converted using the script `notebooks/preprocessing/extract-economic-data.py`. Missing raw datasets are downloaded automatically, except for those of the Belgian National Bank.

+ `conversion_matrices.xlsx` contains conversion matrices to more easily aggregate data from different sector classifications. F.i. converting from NACE 64 to WIOD 55 classification.
+ `census2011_NACE21.csv` contains per Belgian arrondissement (43 in total) the number of employees in every sector of the NACE 21 classification.
+ `others.csv` contains the sectoral output during business-as-usual, household demand during business-as-usual, other final demand during business-as-usual, the desired stock, consumer demand shock, other demand shock, sectoral employees during business-as-usual and sectoral employees under lockdown. Data from various sources. NACE 64 classification.
+ `IO_NACE64.csv` contains the input-output table for Belgium, formatted to NACE 64 classification.
+ `IHS_critical_NACE64.csv` contains the IHS Market Analysts data, reformatted from WIOD 55 to the NACE 64 classification.

#### mobility

Note: only saved locally and on S-drive (not on Git due to NDA). Contains processed origin-destination matrices at the level of municipalities, arrondissements and provinces:
+ staytime: `fractional-mobility-matrix_staytime_*_*.csv`: origin-destination matrices in terms of estimated length of stay, normalised over the total available time. Subtracted 8 hours of sleep per day, and corrected for the GDPR-protected -1 values
+ baseline: `fractional-mobility-matrix_staytime_arr_baseline-*.csv`: normalised origin-destination matrix for three distinct periods: business days, weekends and vacation days. These may be used to distinguish how many additional people were forced to stay at home, and/or to estimate the goal of travel (work/leisure/...)

#### seroprelevance

+ `sero_national_stratified_own` contains for every of the seven sampling waves and for 10 age groups the mean and 95% confidence interval on the seroprelevance of SARS-CoV-2 IGG antibodies. This dataset is made by analysing `~/data/raw/sero/serology_covid19_belgium_round_1_to_7_v20210406.csv` using the script `~/notebooks/preprocessing/herzog-sero-data-analysis.py`. Currently, the positive samples are unweighted, the analysis is performed on the national level and the analysis does not account for the test sensitivity and specificity.
+ `sero_national_overall_herzog.csv` contains for every of the seven sampling waves, the non age-stratified seroprelevance in the Belgian population. Copied from table S1 in the manuscript of Sereina Herzog, available on Medrxiv: https://www.medrxiv.org/content/10.1101/2020.06.08.20125179v5.full-text.

### simulated

Contains zarr directories, which in turn contain groups that each hold a different simulation result. The aim of this 'simulation database' is to be able to save simulation results and perform post-processing without always having to go through the long and computationally demanding task of simulating (using the base.py sim function). This is especially relevant for spatially stratified SEIRD extended models, as these typically take G times longer to run (where G is the level of spatial stratification).

Simulations are saved here using the `utils.py` function `save_sim()` and opened using the `open_sim()` function. The content of the simulations is suggested in the directory titles and mentioned in the 'description' attribute of the zarr groups. It is also printed upon opening the simulation with `open_sim()`. Additionally, the simulation is quickly described here as well.

#### Sanity-check_100sims_100days_Nctot-to-Nchome-day40.zarr

+ `arr_1E-per-arr` description: "Stochastic spatial SEIRD extended model with 100 parallel simulations in 43 arrondissements over 100 days.At day 0 a single exposed person in the age class 30-40 is released ineach of the arrondissements. At day 40 measures are imposed, bringing down the contact rate from Nc_total to Nc_home over the course of 5 + 5 days (tau and l compliance parameters) reaches full effect."