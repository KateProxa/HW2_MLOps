import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Загрузка данных
def load_data(input_path):
    df = pd.read_csv(input_path)
    return df

# Удаление пропущенных значений и дубликатов
def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    return df

# Сохранение
def save_data(df, output_path): 
    df.to_csv(output_path, index=False)

# Основной процесс обработки данных
def main(input_path, output_path):
    df = load_data(input_path)
    df = clean_data(df)

    # Таргет
    df['Obesity_Binary'] = df['NObeyesdad'].apply(lambda x: 1 if x in ['Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I', 'Overweight_Level_II'] else 0)
    df.drop(columns=['NObeyesdad'], inplace=True)

    # Разделение признаков и целевого признака (качество вина)
    features = df.drop(columns=['Obesity_Binary'])
    target = df['Obesity_Binary']

    cat_columns = []
    num_columns = []
    for column_name in features.columns:
        if features[column_name].dtype == object:
            cat_columns.append(column_name)
        else:
            num_columns.append(column_name)

    for col in cat_columns:
        if df[col].nunique() == 2:  # Если всего 2 уникальных значения
            df[col] = df[col].map({df[col].unique()[0]: 0, df[col].unique()[1]: 1})
    
    df['MTRANS_Binary'] = df['MTRANS'].apply(lambda x: 1 if x == 'Public_Transportation' else 0)
    df['CAEC_binary'] = df['CAEC'].apply(lambda x: 1 if x in ['Frequently', 'Always'] else 0)
    df['CALC_binary'] = df['CALC'].apply(lambda x: 1 if x in ['Frequently', 'Always'] else 0)

    df.drop(columns=['MTRANS', 'CALC', 'CAEC'], inplace=True)

    scaler = StandardScaler()
    df[num_columns] = scaler.fit_transform(df[num_columns])

    save_data(df, output_path)


    if __name__ == "__main__":
        main("./data/ObesityDataSet.csv", "./data/clean_data.csv")


