## Data sets for COVID 19 model

### Raw data sets

#### Economic data

+ `Employment - annual detailed data - Domestic concept - A38.xlsx` contains the number of employees per sector from 2014 to 2018. Belgian data. Received from prof. Gert Peersman.
+ `GDP_Belgium_per sector.xlsx` contains the added value at basic prices per sector from 2014 to 2018. Received from prof. Belgian data. Gert Peersman.
+ `input-output.xlsx` contains sectoral input-ouput tables for Belgium. Belgian data. Received from prof. Gert Peersman.
+ `Sectoral_data.xlsx` contains the added value and the number of employees per sector for the year 2018. Contains sectoral social interaction correction factor. Belgian data. Received from prof. Gert Peersman.
+ `Staff distribution by sector.xlsx` contains the fraction of employees who worked at the workplace, at home, in a mix of both, those temporarely unemployed and those abscent during the Belgian lockdown of March 17th, 2020 to March 4th, 2020. Belgian data. Received from prof. Gert Peersman.
+`Supply and use table - Belgium.xlsx` contains the sectoral input-output tables for 2016. Belgian data. Received from prof. Gert Peersman.

#### Hospital data

+ `symptomOnsetHospitalization.xlsx` contains: 1) the date at which patients first reported having symptoms, 2) the data at which patients were hospitalized and 3) the age of the patient. Received from Ghent University hospital, contact: Pascal Coorevits.

+ `AZmariaMiddelares.xlsx` contains: 1) patient ID, 2) per patient: in chronological order, from bottom to top (!), the amount of time spent in the emergency room, cohort or intensive care unit, 3) the age of the patient, 4) the sex of the patient, 5) if the patient recovered or died. 'cohortafdeling D601' is a geriatric cohort ward, 'cohortafdeling D501' is a regular cohort ward. Received from AZ Maria Middelares, contact: Leen Van Hoeymissen.

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

+ `verity_etal.csv` contains age-stratified estimates of the number of symptomatic cases which result in hospitalization, the number of hospitalized patients in need of intensive care and the case fatality ratio. Data were obtained from https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext
+ `wu_etal.csv` contains age-stratified estimates of the fraction of asymptomatic cases. Data were obtained from https://www.nature.com/articles/s41591-020-0822-7
+ `others.csv` contains all other parameters used to run the model. Obtained from various sources, detailed in the report.
+ `split_mortality.csv` contains age-stratified probability to die after infection,
split between general population and elderly homes. Data from https://m.standaard.be/cnt/dmf20200609_04985767.
+ `non_stratified.csv` contains non-stratified estimates of asymptomatic cases, symptomatic cases which result in hospitalization, hospitalized patients in need of intensive care and the case fatality ratio

### Interim data sets conversion scripts

Conversion scripts are managed inside the `covid19model` package (`src/covid19model/data` folder).

- ...
