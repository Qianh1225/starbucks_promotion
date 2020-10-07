# Starbucks Promotion Strategy
This project is a part of Udacity Data Science Nanodegree

### 1. Project Overview<a name="overview"></a>

Starbucks provided simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer tousers. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. If the customer finish the requirement before offer expired, they will receive the reward.  The goal of the promotion is to motivate individuals to make more purchases and increase the profits for the company. Therefore, our objective will be to identify customers whom are inclined to spent more when offered with the promotion. We will define this type of customer as 'responsive' to a specific offer. Our plan is to design a ML model which will predict whether a customer will spent more on a specific offer given the customer's information.


### 2. Project Components<a name="components"></a>

The problem that I chose to solve was to build a model that predicts whether a customer will viewed an offer and spent more than normal.
Here are the 3 notebooks that I am working on through the analysis :  

1- data cleaning.ipynb

2- Feature engineer and EDA.ipynb

3- modelling.ipynb

4- model_helper.py

### 3. Installation<a name="installation"></a>

 - The code should run with no issues using Python versions 3.*.
 - No extra besides the built-in libraries from Anaconda needed to run this project
 - Data Processing & Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
 - Data Visualization: Matplotlib, Seaborn

### 4. File Descriptions<a name="files"></a>

The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

### 5. Instructions<a name="instructions"></a>

1. The entire anlaysis is contained within the jupyter notebook.
2. All 3 json files should be located in data folder.


### 6. Acknowledgements<a name="licensing">

This project was completed as part of the [Udacity Data Science Nanodegree]. The dataset used in this project contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. [Starbucks® Rewards program: Starbucks Coffee Company](https://www.starbucks.com/rewards/).

### 7. Results<a name="results"></a>
The main observations of the code are published on medium [here]()
