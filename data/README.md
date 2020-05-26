## Data sets for COVID 19 model

### Raw data sets

#### Economic data

+ `Employment - annual detailed data - Domestic concept - A38.xlsx` contains the number of employees per sector from 2014 to 2018. Belgian data. Received from prof. Gert Peersman.
+ `GDP_Belgium_per sector.xlsx` contains the added value at basic prices per sector from 2014 to 2018. Received from prof. Belgian data. Gert Peersman.
+ `input-output.xlsx` contains sectoral input-ouput tables for Belgium. Belgian data. Received from prof. Gert Peersman.
+ `Sectoral_data.xlsx` contains the added value and the number of employees per sector for the year 2018. Contains sectoral social interaction correction factor. Belgian data. Received from prof. Gert Peersman.
+ `Staff distribution by sector.xlsx` contains the fraction of employees who worked at the workplace, at home, in a mix of both, those temporarely unemployed and those abscent during the Belgian lockdown of March 17th, 2020 to March 4th, 2020. Belgian data. Received from prof. Gert Peersman.
+`Supply and use table - Belgium.xlsx` contains the sectoral input-output tables for 2016. Belgian data. Received from prof. Gert Peersman.

#### UZ Ghent data

+ `symptomOnsetHospitalization.xlsx` contains: 1) the date at which patients first reported having symptoms, 2) the data at which patients were hospitalized and 3) the age of the patient. Received from Ghent University hospital, contact: Pascal Coorevits.

#### Polymod

+ `contacts.Rdata` contains the characteristics of 97,904 human contacts made by 9,000 participants during one day. Data were gathered during the 2008 Polymod study by Mossong. The characterstics including age, sex, location, duration, frequency, and occurrence of physical contact. The data was downloaded from https://lwillem.shinyapps.io/socrates_rshiny/.
+ `Age pyramid of Belgium.csv` contains the age pyramid of Belgium. Given as the number of individuals per gender and per 5 year age bins. Retreived from https://statbel.fgov.be/en/themes/population/structure-population

#### Google

+ `community_mobility_data.csv` contains a copy of the Google Community Mobility Report dataset downloaded by the function `get_google_mobility_data()`. Mobility data is extracted from 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=2dcf78defb92930a'

#### Others

+ `imperialCollegeAgeDist.csv` contains age-stratified estimates of the number of symptomatic cases which result in hospitalization, the number of hospitalized patients in need of intensive care and the infection fatality ratio. Data were obtained from https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext

### Interim data sets conversion scripts

Conversion scripts are managed inside the `covid19model` package (`src/covid19model/data` folder).

- ...