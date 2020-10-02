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

#### Polymod

+ `contacts.Rdata` contains the characteristics of 97,904 human contacts made by 9,000 participants during one day. Data were gathered during the 2008 Polymod study by Mossong. The characterstics including age, sex, location, duration, frequency, and occurrence of physical contact. The data was downloaded from https://lwillem.shinyapps.io/socrates_rshiny/.
+ `Age pyramid of Belgium.csv` contains the age pyramid of Belgium. Given as the number of individuals per gender and per 5 year age bins. Retreived from https://statbel.fgov.be/en/themes/population/structure-population

#### Google

+ `community_mobility_data.csv` contains a copy of the Google Community Mobility Report dataset downloaded by the function `get_google_mobility_data()`. Mobility data is extracted from https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=2dcf78defb92930a

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

### Interim data sets

Conversion scripts are managed inside the `covid19model` package (`src/covid19model/data` folder).

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

#### Demographic data

+ `age_structure_per_arrondissement.csv` : population of each age per arrondissement

+ `age_structure_per_municipality.csv` : population of each age per municipality

+ `age_structure_per_province.csv` : population of each age per province

+ `area_arrond.csv` contains the area of Belgian arrondissements per NIS code in square meters

+ `area_municip.csv` contains the area of Belgian municipalities per NIS code in square meters

+ `area_province.csv` contains the area of Belgian provinces per NIS code in square meters

+ `initN_arrond.csv` contains a pandas dataframe with the following columns: arrondissement NIS-code, total population, population aged 0-9, population aged 10-19, ..., population aged 80 and above. Created in `notebooks/JV-extract-age-structures.ipynb`.

+ `initN_province.csv` contains a pandas dataframe with the following columns: province NIS-code, total population, population aged 0-9, population aged 10-19, ..., population aged 80 and above. Created in `notebooks/JV-extract-age-structures.ipynb`.

+ `initN_municip.csv` contains a pandas dataframe with the following columns: municipality NIS-code, total population, population aged 0-9, population aged 10-19, ..., population aged 80 and above. Created in `notebooks/JV-extract-age-structures.ipynb`.

#### Economic data

All economic data from the data/raw/economical was converted in the notebook `notebooks/twallema-extract-economic-data.ipynb`.

+ `conversion_matrices.xlsx` contains conversion matrices to more easily aggregate data from different sector classifications. F.i. converting from NACE 64 to WIOD 55 classification.
+ `census2011_NACE21.csv` contains per Belgian arrondissement (43 in total) the number of employees in every sector of the NACE 21 classification.
+ `others.csv` contains the sectoral output during business-as-usual, household demand during business-as-usual, other final demand during business-as-usual, the desired stock, consumer demand shock, other demand shock, sectoral employees during business-as-usual and sectoral employees under lockdown. Data from various sources. NACE 64 classification.
+ `IO_NACE64.csv` contains the input-output table for Belgium, formatted to NACE 64 classification.
+ `IHS_critical_NACE64.csv` contains the IHS Market Analysts data, reformatted from WIOD 55 to the NACE 64 classification.
