
## Investigating a Dataset - Titanic Data

Data was obtained from the Kaggle website: [Titanic Data](https://www.kaggle.com/c/titanic/data)

### Question(s): What variables could have strong relationship on Survival?
### What variables increases your chances of not surviving the ship wreck?


### Imports:


```python
#import all necessary packages
%pdb 
%matplotlib inline                 
import pandas as pd                 #pd is a convention
import numpy as np                  #np is a convention
import matplotlib.pyplot as plt    #plt is a convention
import seaborn as sns              #Improves visualizations
from IPython.core.interactiveshell import InteractiveShell            #Allows value of multiple statements at once
InteractiveShell.ast_node_interactivity = "all"
```

    Automatic pdb calling has been turned ON


### Variable Descriptions:

   ### Variable      Definition            Key
    survival      Survival         0 = No, 1 = Yes

    pclass      Ticket class     1 = 1st, 2 = 2nd, 3 = 3rd

    sex          Sex

    Age         Age 

    sibsp      # of siblings / spouses aboard the Titanic
    
    parch      # of parents / children aboard the Titanic	

    ticket     Ticket number
    
    fare       Passenger fare

    cabin      Cabin number

    embarked   Port of Embarkation	C = Cherbourg, Q = Queenstown, S=Southampton

### Variable Details:
**pclass**: A proxy for socio-economic status (SES)

1st = Upper

2nd = Middle

3rd = Lower

**age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

**sibsp**: The dataset defines family relations in this way...

Sibling = brother, sister, stepbrother, stepsister

Spouse = husband, wife (mistresses and fiancés were ignored)

**parch**: The dataset defines family relations in this way...

Parent = mother, father

Child = daughter, son, stepdaughter, stepson

Some children travelled only with a nanny, therefore parch=0 for them.


### Variable Classification:

***Dependent variable:*** Survived

***Independent variable's:*** Pclass, Gender, and Fare

### Importing Data:


```python
titanic_data = pd.read_csv('titanic_data.csv')      #CSV file uploaded to a DataFrame
titanic_data.head()                                 # prints out the first 5 lines
titanic_data.tail()                                 # prints out the last 5 lines
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.00</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.00</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.45</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.00</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.75</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_data.shape                #891 rows and 12 columns

print('Data Types:')
titanic_data.dtypes               #data types in each column

print('Contents of Data Set:')
titanic_data.info()                #The contents of the data set
```




    (891, 12)



    Data Types:





    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object



    Contents of Data Set:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB



```python
def missingvaluecol(data):                            #Creating a function to find NAN's in each column
    return sum(data.isnull())

print('Missing Values in each Column:')
titanic_data.apply(missingvaluecol, axis=0)

```

    Missing Values in each Column:





    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64



We have created a function to apply to each column of the DataFrame to find missing values.
We found that there are 177 passenger age's missing, 687 passenger cabin numbers missing, and 2 port of embarkation missing.

Above we printed out the first 5 rows and their respective columns of the Titanic Data set, based on the shape of the data set there are 891 passengers and 12 columns of criteria. We also see that the variables have different data types such as Integers, Objects,and Floats.

To understand the data provdided we apply the describe function to the DataFrame to obtain summary statistics.

When trying to obtain the stats from each column we noticed that the missing values from the Age column were causing a problem when trying to get the quartile's. The 'na' values from the Age column were removed along with their corresponding rows, we do not have a good basis to imput the data.


A salient point is that the Sex column is an object. We would to try to convert this into a numeric value if we want to use to further investigate. We attribute male to a value of 1 and female to a value of 0.


```python
titanic_data['Gender'] = np.where(titanic_data['Sex'] == 'male', 1, 0) #A new variable based on Gender
```


```python
titanic_data['Survival'] = titanic_data.Survived.map({0 : 'Died', 1 : 'Survived'})  #Get use to providing more descriptive labels, attaching a new series to the Dataframe
titanic_data['Class'] = titanic_data.Pclass.map({1 : 'Upper Class', 2 : 'Middle Class', 3 : 'Lower Class'})     #dataframe.series.map - map values of series
```


```python
titanic_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Gender</th>
      <th>Survival</th>
      <th>Class</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
      <td>Died</td>
      <td>Lower Class</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>0</td>
      <td>Survived</td>
      <td>Upper Class</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>0</td>
      <td>Survived</td>
      <td>Lower Class</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>0</td>
      <td>Survived</td>
      <td>Upper Class</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
      <td>Died</td>
      <td>Lower Class</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_data.dropna(subset=['Age'],inplace=True)  #Dropped the rows that had NAN'S in the "Age"column
```

We also noticed that the PassengerID, Cabin, and Ticket columns can be removed from the Titanic Data Set, it will help our code run faster.


```python
titanic_data.drop(['PassengerId','Cabin','Ticket'],axis=1, inplace=True)  #Delete's PassengerID,Cabin, and Ticket column inplace
```


```python
titanic_data.head()          #Print first 5 rows to confirm that the columns were deleted appropriately
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Gender</th>
      <th>Survival</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
      <td>Died</td>
      <td>Third Class</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>0</td>
      <td>Survived</td>
      <td>Upper Class</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>0</td>
      <td>Survived</td>
      <td>Third Class</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>0</td>
      <td>Survived</td>
      <td>Upper Class</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
      <td>Died</td>
      <td>Third Class</td>
    </tr>
  </tbody>
</table>
</div>



### Statistics of each column:


```python
titanic_data.describe() #provides summary statistics without the NA values in 'Age' column
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
      <td>0.647587</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
      <td>0.477990</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Above we have trimmed the data removing rows that contain NAN's in the Age column. We notice that the data now contains 714 passengers. 


The Age column indicates that on average the passengers were about 29.70 years old. The median value that is shown in the (50 percent quartile) indicates that the Age is 28 years old of the passengers. We also notice that the youngest passenger aboard is .42 less than a year old and the oldest passenger aboard is 80 years old.

The Fare column that provides information on the price paid to board the Titanic shows that on average the passengers paid about 34 shillings. We also notice that standard deviation is large showing how close the relative data points are from the mean, the std shows that it is 52.91 standard deviations meaning that the values are spread out over a wider range of values. We wouldn't use the mean to get a bearing on the average passenger fare since it is susceptible to extreme values, instead we will look at the median(50% quartile) this shows that passengers paid about 15.74 shillings to board the ship. We also note the disparity in the price range of the fare by looking at the minimum passenger fare being 0 and the maximum passenger fare is about 512.33 shillings. It is safe to assume that the passengers that didn't pay were probably working or toddlers. The 512.33 shillings was probably paid by someone pertaining to the upper-class.



### Bar graph:


```python
titanic_data['Count'] = 1        #attaches Series Count to df to ensure every row is counted once 
countClass = titanic_data.groupby(['Class']).count()['Count']  #groupby method groups by the categories in the variable, that it is call on, the count at the end just selects variable from the created dataframe   
passengers_class = countClass.plot(kind='bar')
passengers_class.set_title('Passengers In Each Economic Class')
passengers_class.set_ylabel('Passengers')
print countClass
```




    <matplotlib.text.Text at 0x115d8d510>






    <matplotlib.text.Text at 0x1159634d0>



    Class
    Lower Class     491
    Middle Class    184
    Upper Class     216
    Name: Count, dtype: int64



![png](output_25_3.png)


We have provided two bar charts that clearly depict the amount of passengers that belong to each Economic class and whether they survived the ship wreck. The counts are provided above: most of the passengers belonged to the lower class and did not survive the ship wreck. 

We will proceed by creating a scatter plot of each of the continuous variables within the data set to see if there is any noticeable relationships. This will be done by examining the associations with Fare and Age. This will help us determine if there is any superflous relationships with the variables of interest. 

### Scatter Plot:


```python
scatterplot_fareage = titanic_data.plot(kind='scatter', x='Fare', y='Age')
scatterplot_fareage.set_title('Scatter Plot of Fare and Age Variables')
```




    <matplotlib.text.Text at 0x116368210>




![png](output_29_1.png)


We notice that the majority of passengers paid an amount less than 100 shillings, there is not a clear relationship between these variables. This can probably be a result of passengers coming with their parents who would be responsible to pay for the fare.

### Correlations:

Next we see if any variables are related by creating a correlation matrix to inspect the relationships between the variables at once. 


```python
titanic_data[['Age','Fare',]].corr()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.000000</td>
      <td>0.096067</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.096067</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



We have created a correlation matrix that omits the categorical variables within the dataset. We have used Pearson's correlation method to compute these statistics. Based on the scatterplot these results corroborate that there is a weak relationship between the age of the passengers and the amount paid to embark the ship. 


```python
pd.crosstab(titanic_data["Pclass"],titanic_data["Survived"],margins=True)
pd.crosstab(titanic_data["Fare"],titanic_data["Survived"],margins=True)
pd.crosstab(titanic_data["Gender"],titanic_data["Survived"],margins=True)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Survived</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>64</td>
      <td>122</td>
      <td>186</td>
    </tr>
    <tr>
      <th>2</th>
      <td>90</td>
      <td>83</td>
      <td>173</td>
    </tr>
    <tr>
      <th>3</th>
      <td>270</td>
      <td>85</td>
      <td>355</td>
    </tr>
    <tr>
      <th>All</th>
      <td>424</td>
      <td>290</td>
      <td>714</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Survived</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Fare</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>6</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4.0125</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6.2375</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6.4375</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6.45</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6.4958</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6.75</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6.975</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7.0458</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7.05</th>
      <td>6</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7.0542</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7.125</th>
      <td>4</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7.1417</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7.225</th>
      <td>4</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7.2292</th>
      <td>6</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7.25</th>
      <td>9</td>
      <td>1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>7.4958</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7.5208</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7.55</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7.65</th>
      <td>3</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7.7333</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7.7417</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7.75</th>
      <td>10</td>
      <td>4</td>
      <td>14</td>
    </tr>
    <tr>
      <th>7.775</th>
      <td>12</td>
      <td>2</td>
      <td>14</td>
    </tr>
    <tr>
      <th>7.7958</th>
      <td>4</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7.8</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7.8542</th>
      <td>10</td>
      <td>3</td>
      <td>13</td>
    </tr>
    <tr>
      <th>7.875</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7.8792</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>80.0</th>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>81.8583</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82.1708</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>83.1583</th>
      <td>0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>83.475</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>86.5</th>
      <td>0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>89.1042</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>90.0</th>
      <td>1</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>91.0792</th>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>93.5</th>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>106.425</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>108.9</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>110.8833</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>113.275</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>120.0</th>
      <td>0</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>133.65</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>134.5</th>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>135.6333</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>146.5208</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>151.55</th>
      <td>2</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>153.4625</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>164.8667</th>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>211.3375</th>
      <td>0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>211.5</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>227.525</th>
      <td>0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>247.5208</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>262.375</th>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>263.0</th>
      <td>2</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>512.3292</th>
      <td>0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>All</th>
      <td>424</td>
      <td>290</td>
      <td>714</td>
    </tr>
  </tbody>
</table>
<p>221 rows × 3 columns</p>
</div>






<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Survived</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>64</td>
      <td>197</td>
      <td>261</td>
    </tr>
    <tr>
      <th>1</th>
      <td>360</td>
      <td>93</td>
      <td>453</td>
    </tr>
    <tr>
      <th>All</th>
      <td>424</td>
      <td>290</td>
      <td>714</td>
    </tr>
  </tbody>
</table>
</div>



We notice that based on Pclass we see that the majority of upper class passengers did not survived. This could be that during these times they were given preferential treatment exposing them to a riskier area. We also note that the most people survived belonged to the lower class. 

We also notice that based on Gender we see that the majority of survivors were Females. This will probably be because of the given times of the incident meaning men would be the last ones to evacuate the ship.

### Logit Model:

We will now construct a logistic model with the variables that have the strongest relationship with the dependent variable. The dependent variable in this model would be Survived and the independent variables would be Pclass, Gender, and Fare. The choice for a logistic model is that the dependent variable is categorical. 


```python
import statsmodels.api as sm
import pylab as pl
```

We first create dummy variables for class and gender in order to control for these and show the impact of a particular class or gender on the chances of surviving.


```python
dummies_class = pd.get_dummies(titanic_data['Pclass'], prefix='Pclass')
print dummies_class.head()
dummies_gender = pd.get_dummies(titanic_data['Gender'], prefix='Gender')
print dummies_gender.head()
```

       Pclass_1  Pclass_2  Pclass_3
    0         0         0         1
    1         1         0         0
    2         0         0         1
    3         1         0         0
    4         0         0         1
       Gender_0  Gender_1
    0         0         1
    1         1         0
    2         1         0
    3         1         0
    4         0         1


Now before we proceed we must create a dataframe that is going to contain the dummy variables,the continous variable Fare, and the dependent variable Survived. 


```python
columns_reg = ['Survived','Fare']
dat = titanic_data[columns_reg].join(dummies_class.ix[:, 'Pclass_2':]) #we start at the second column to aviod mulit-colinearity
dat['Intercept'] =1.0                          #adding an intercept 
data1 = dat.join(dummies_gender.ix[:,'Gender_1':])
data1.head()                            #Included the Gender_1 column that contains dummies for Gender 
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Fare</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Intercept</th>
      <th>Gender_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>71.2833</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>7.9250</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>53.1000</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
columns = data1.columns[1:]
logit = sm.Logit(data1['Survived'],data1[columns])
model = logit.fit()
model.summary()
```

    Optimization terminated successfully.
             Current function value: 0.470231
             Iterations 6





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>Survived</td>     <th>  No. Observations:  </th>  <td>   714</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   709</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>     4</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Wed, 19 Apr 2017</td> <th>  Pseudo R-squ.:     </th>  <td>0.3038</td>  
</tr>
<tr>
  <th>Time:</th>              <td>19:25:22</td>     <th>  Log-Likelihood:    </th> <td> -335.74</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -482.26</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>3.459e-62</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>Fare</th>      <td>    0.0022</td> <td>    0.002</td> <td>    0.945</td> <td> 0.345</td> <td>   -0.002     0.007</td>
</tr>
<tr>
  <th>Pclass_2</th>  <td>   -0.8005</td> <td>    0.289</td> <td>   -2.768</td> <td> 0.006</td> <td>   -1.367    -0.234</td>
</tr>
<tr>
  <th>Pclass_3</th>  <td>   -1.8308</td> <td>    0.281</td> <td>   -6.509</td> <td> 0.000</td> <td>   -2.382    -1.280</td>
</tr>
<tr>
  <th>Intercept</th> <td>    2.1501</td> <td>    0.306</td> <td>    7.029</td> <td> 0.000</td> <td>    1.551     2.750</td>
</tr>
<tr>
  <th>Gender_1</th>  <td>   -2.5529</td> <td>    0.204</td> <td>  -12.494</td> <td> 0.000</td> <td>   -2.953    -2.152</td>
</tr>
</table>



### Tentative Results:

The logit regression model above was estimated using Maximum Likelihood Estimation(MLE) using the available data to form parameters for each of the variables. We see that the output table provides us the coefficients for the model associated with each variable. The z-score gives us an idea of how well the coefficients fit into the model. The overall fit of the model is given by the Psuedo R-square the value is .3038, 1 is the upper bound meaning that the independent variables are perfectly explaining the dependent variable. If R-square approaches 1 this means that the values of the data points converging to the regression line. 

We notice that there is an inverse relationship between the probability of surviving and the economic class that a passenger belongs to. The probability of surviving the Titanic Ship wreck based on this data is higher for passengers that belong to the higher class and it lowers as you descend classes Middle classes and Lower classes holding other variables constant. If we could have included a variable that would segment the ship into different areas where the passengers of a particular class were located, this could raise the statistical robustness of our variables. We also notice that there is an inverse relationship between the probability of surviving and the Gender a passenger pertains to. The probability of surviving the Titanic Ship wreck based on this data is lower for passengers that are males holding other variables constant.

We also take a look at the t-stat for the coefficients within the model. We notice that t-stat for Gender_1 is 12.494 meaning that it is statistically significant at the 1 percent level. This gives us a good reason supporting the creation of dummy variables for this categorical variable to include in the model. We also notice that the 2 dummy variables for classes are statistically significant Middle class 2.768 and Lower Class 6.509 these are both statistically significant at the 1 percent level. We also notice that the Fare variable is not statistically significant at 1, 5, 10 percent level. 





Based on these results we will briefly provide an explanation on the relationships on the independent variables to the dependent variables. We notice that class has an inverse relationship with Survived. We also notice that Gender has an inverse relationship with Survived. We also note that Fare has positive relationship with Survived. If you pertained to the middle class you had less of a chance to survive. We also notice that you also have even less of a chance if you belonged to the lower class. This is made apparent based on the coefficients of the independent varibles in the model. If you purchased an expensive boarding pass you have a higher chance of surviving. If you are a male you have less of a chance surviving the ship wreck, this can be said by looking at the negative sign of the coefficient for the Gender_1 variable. 

### Limitations:


As we noted earlier the dataset contains missing values in the Age column, we have ommited the corresponding rows(passengers). This eliminates 177 passengers that have their age missing, this can hinder our analysis this can skew the data points into a particular direction. There could have been other potential variables, proxy variables that could have been used in the logit model to reduce biasness from each of the independent variables. They could have been used as control variables, for instance a variable that divided the ship into 4 different regions. 
