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
# MONDAY 11.12
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

################################################################################
# MONDAY 11.26
################################################################################

#PROBLEM 3. There's a data quality issue hiding in the admissions dataset from Monday. Correct this issue and compare your new results. How are they the same? How do they differ?

#The issue was that there was a space after male for a few of the spaces. This resulted in skewed results because it only picked up the "male" without a space (only 2 of them instead of 7 of them). A way to correct this is either to edit the CSV itself or make the men = df[(df['Gender] == 'Male)] also equal to 'Male '. For the question "Does department correlate with admission", both results stayed the same, however for the t-test (which is the statistic and p-value) changed with the altered data. 

#PROBLEM 4. The data also represents an example of Simpson's Paradox. Use whatever visualization tools you'd like to illustrate the two possible perspectives. Make sure to include a screenshot of each and explain the perspective shown in each. 



#Some scholars contend that Shakespeare's early plays are actually collaborations that could be attributed to other authors. One such author is Christopher Marlow (1564-1593). Use a series of visualizations to compare data from Marlowe's plays and Shakespeare's plays. The zip file in Assignment Data contains CSV data looking at both sentiment and word counts comparing plays from the two authors. Use the data and Altair to work through the following problems (submit your notebook with additional documentation addressing each question): 

#PROBLEM 5. Build a visualization that allows you to compare the distribution of positive sentiment in both Marlowe and Shakespeare. What does this tell you about the styles of the two authors?

#PROBLEM 6. Build a visualization that allows you to explore the correlation of word counts for Marlowe and Shakespeare. What does this tell you about the styles of the two authors? 

#PROBLEM 7. Generate three additional visualizations using this data (please use different visualization techniques for each visualization). What do these visualizations tell you about potential collaborations between Shakespeare and Marlowe? 