
def process_missing(df):
    """Handle various missing values from the data set

    Usage
    ------

    holdout = process_missing(holdout)
    """
    df["Fare"].fillna(train["Fare"].mean(), inplace=True)
    df["Embarked"].fillna("S", inplace=True)
    return df

def process_age(df):
    """Process the Age column into pre-defined 'bins' 

    Usage
    ------
    train = process_age(train)
    """
    df["Age_categories"] = pd.cut(
        df["Age"].fillna(-0.5),
        [-1,0,5,12,18,35,60,100],
        labels=[
            "Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"
        ]
    )
    return df

def process_fare(df):
    """Process the Fare column into pre-defined 'bins' 

    Usage
    ------

    train = process_fare(train)
    """
    df["Fare_categories"] = pd.cut(df["Fare"], bins=[-1, 12, 50, 100, 1000],
                                   labels=["0-12", "12-50", "50-100", "100+"])
    return df

def process_cabin(df):
    """Process the Cabin column into pre-defined 'bins'

    Usage
    ------

    train process_cabin(train)
    """
    df["Cabin_type"] = df["Cabin"].str[0].fillna("Unknown")
    df = df.drop('Cabin', axis=1)
    return df

def process_titles(df):
    """Extract and categorize the title from the name column 

    Usage
    ------

    train = process_titles(train)
    """
    titles = {
        "Mr" :         "Mr",
        "Mme":         "Mrs",
        "Ms":          "Mrs",
        "Mrs" :        "Mrs",
        "Master" :     "Master",
        "Mlle":        "Miss",
        "Miss" :       "Miss",
        "Capt":        "Officer",
        "Col":         "Officer",
        "Major":       "Officer",
        "Dr":          "Officer",
        "Rev":         "Officer",
        "Jonkheer":    "Royalty",
        "Don":         "Royalty",
        "Sir" :        "Royalty",
        "Countess":    "Royalty",
        "Dona":        "Royalty",
        "Lady" :       "Royalty"
    }
    extracted_titles = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
    df["Title"] = extracted_titles.map(titles)
    return df

def create_dummies(df, column_name):
    """Create Dummy Columns (One Hot Encoding) from a single Column"""
    return pd.concat([df, pd.get_dummies(df[column_name], prefix=column_name)], axis=1)
