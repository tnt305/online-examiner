"""
### Third stage approach

https://www.isixsigma.com/hypothesis-testing/hypothesis-testing-fear-no-more/
https://github.com/eceisik/eip/blob/main/hypothesis_testing_examples.ipynb

Create a hypothesis test based on the cheat logs results
- Cheat logs results should have features: time spend + action detected

What we want to do is for suspected case, can he/she considered to break the exam regulations or not.
At this stage, i proposed two hypo test that we can consider:
1. Supposing time consuming on the screen on average >= 70% time of the test means trustable, 
the real-time consuming for example is 65% can be considered violation ?

2. If action 1 and action 2 shares the same time spent on, can we say they
"""
import pandas as pd
#import scikit_posthocs as sp
import numpy as np
from scipy import stats
from scipy.stats import ttest_1samp
import statsmodels
from statsmodels.stats.weightstats import ztest as ztest
import pingouin as pg

pd.options.display.float_format = '{:,.4f}'.format


def check_normality(data , p_value):
    _, p_value_normality=stats.shapiro(data)
    return p_value_normality  
def check_variance_homogeneity(group1, group2):
    _, p_value_var= stats.levene(group1,group2)
    return p_value_var

def work_time_hypotest(log_file, total_exam_duration, ratio = 0.7, max_duration = 3, p = 0.05):
    '''
    param: log_file : the csv/txt path recordings of duration of each action 
    param: total_exam_duration : the time in seconds, which the candidate spends on the test/exam
    param: the percentage of exam time that candidates must look into the camera
    param: max_duration is the maximum test time in second or mins
    param: p_value can change 0.01, 0.05 or 0.1 in case of extend/minor the condition
    '''
    df = pd.read_csv(log_file)
    duration = np.array(df['duration'][df['cheat_type'].isin(['lookInto', 'neutral'])])
    cheat_type = df['cheat_type']

    if check_normality(duration, p) < p:
        print('data not normally distributed, chi-quare test')
        t_statistic, p_value = ztest(duration, ratio*total_exam_duration/len(duration), alternative= 'larger')
        if p_value/2 <= 0.5:
            print('Not sure that the candidate is cheating')
        else:
            print('Cheater detected')
    else:
        print('data normally distributed, t-test taking')
        t_statistic, p_value = ttest_1samp(duration, ratio*total_exam_duration/len(duration), alternative = 'greater')
        if p_value/2 <= 0.5:
            print('Not sure that the candidate is cheating')
        else:
            print('Cheater detected')

def cheat_pair_hypotest(log_file, p = 0.05, action = 'LookingLeft', reversed_action = 'LookingRight', p = 0.05):
    '''
    We take the test between cases with actions that supposing opposed to each other
    For example: if a person is looking right as his usual action but the frequency and time of looking left is similar,
    then cheater 
    '''
    df = pd.read_csv(log_file)
    action1 = np.array(df['duration'][df[action]])
    action2 = np.array(df['duration'][df[reversed_action]])
    # action_n can be added if assumptions are raised
    if check_variance_homogeneity(action1, action2) > p:
        print('variance same')
        F, p_value = stats.f_oneway(action1, action2)
        if p_value >p:
            print('Cheater detected')
    else:
        print('variance diff')
        result = pg.welch_anova(dv='score', between='group', data=df)
        if result['p-ucn'] >= p:
            print('Cheater detected')




