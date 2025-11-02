def clean_data(df):
    # Example cleaning operations
    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing values with the mean for numeric columns
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        df[column].fillna(df[column].mean(), inplace=True)

    # Convert categorical columns to category type
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype('category')

    return df

def transform_data(df):
    # Example transformation operations
    # Normalize numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    return df