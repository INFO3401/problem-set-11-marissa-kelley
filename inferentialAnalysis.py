import numpy as np
import pandas as pd
import scipy
#scipy scientific computing package for statistical tests 

import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Extract the data. Return both the raw data and dataframe
def generateDataset(filename):
    data = pd.read_csv(filename)
    df = data[0:]
    df = df.dropna()
    return data, df

################################################################################
# MONDAY 
################################################################################
#I worked with Hannah Weber and Taylor Lawrence 

#PROBLEM 1. What statistical test would you use for the following scenarios? 

#(a) Does a student's current year (e.g., freshman, sophomore, etc.) effect their GPA?
    #The independent variable: Current Year - Categorical 
    #The dependent variable: GPA - Continous  
    #Statistical Test: T-Test

#(b) Has the amount of snowfall in the mountains changed over time? 
    #The independent variable: Amount of snowfall - Continous 
    #The dependent variable: Time - Continous  
    #Statistical Test: Generalized Regression 

#(c) Over the last 10 years, have there been more hikers on average in Estes Park in the spring or summer? 
    #The independent variable: Season (Spring or summer ) - Categorical 
    #The dependent variable: Average # of Hikers - Continous  
    #Statistical Test: T-test

#(d) Does a student's home state predict their highest degree level?
    #The independent variable: Home state - Categorical 
    #The dependent variable: Degree Level - Categorical   
    #Statistical Test: Chi-Squred test 

#PROBLEM 2. You've been given some starter code in class that shows you how to set up ANOVAs and Student's T-Tests in addition to the regression code from the last few weeks. Now, use this code to more deeply explore the simpsons_paradox.csv dataset. Compute new dependent variables that shows the percentage of students admitted and rejected for each row in the CSV. Use those rows to try to understand what significant correlations exist in this data. What factors appear to contribute most heavily to admissions? Do you think the admissions process is biased based on the available data? Why or why not?

# Run a t-test
def runTTest(ivA, ivB, dv):
    #ivA: Indepedent variable for condition A
    #ivB: Independent variable for condition B
    #dv: Dependent variable
    ttest = scipy.stats.ttest_ind(ivA[dv], ivB[dv])
    #Run an Independent sample's ttest, give me the DV values associated with ivA and ivB
    #Use ttest to compare condition A and condition B for some specified DV
    print(ttest)

#Run ANOVA
def runAnova(data, forumula):
    model = ols(formula, data).fit()
    #create a model using ordinal lease squares forumlation using a formula and my data
    #predescribed relationship between DV and how they allow me to predict my IV, so I want to adjust weights in model so that
    #I minimize the total sum squared error 
    aov_table = sm.stats.anova_lm(model, typ = 2)
    #how much variance is explained. statsitical model
    #Type 2 says run a two-way anova- compare whether there's a difference in the model and if there's a difference in the other direction as well
    print(aov_table)
    
#Run the analysis 
#Extract the data
rawData, df = generateDataset('simpsons_paradox.csv')
#will return the rawData and a formated dataframe (df part)

print("Does gender correlate with admissions?")
#IV: Gender 
#DV: Admissions (A-F) - Categorical 
#Stat test: T-test
men = df[(df['Gender'] == 'Male')]
women = df[(df['Gender'] == 'Female')]
#give me all the rows that represent male applications
runTTest(men, women, 'Admitted') #column with admissions counts 

#test stat result: 5.332277756733584
#p value result: 0.001774285663548817 
#Suggest that there is a significant correlation here because p is less than 0.5.

    
print("Does department correlate with admissions?")
simpleFormula = 'Admitted ~ C(Department)'
runAnova(rawData, simpleFormula)
#IV: Department- Categorical
#DV: Admissions (A-F) - Categorical 
#Stat test: Anova 
#Not a significant correlation. P > 1 so there's no significant correlation 

print("Do gender AND department correlate with admissions?")
moreComplex = 'Admitted ~ C(Department) + C(Gender)'
runAnova(rawData, moreComplex)

#DV: Admitted
#IV: Department
#IV: 

#Both have p-values of less than 