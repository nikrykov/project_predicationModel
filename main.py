from writeinflux import WriteInflux
from statsmodels.tsa.ar_model import AutoReg
from matplotlib import pyplot
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import datetime

def print_hi(name):
    print(f'Hi, {name}')

def record(all_df, measurement):
    for index, row in all_df.iterrows():
        json = [
            {
                "measurement": measurement,
                "time": index,
                "fields": {
                    "RMSI_0_0": row["RMSI_0_0"],
                    "RMSV_3_0": row["RMSV_3_0"],
                    "RMSV_2_0": row["RMSV_2_0"],
                    "RMSI_0_1": row["RMSI_0_1"],
                    "RMSV_2_1": row["RMSV_2_1"],
                    "RMSI_0_2": row["RMSI_0_2"],
                    "RMSV_2_2": row["RMSV_2_2"],
                    "RMSI_1_0": row["RMSI_1_0"]
                }
            }
        ]
        object1 = WriteInflux(json)

if __name__ == "__main__":

    print('1. Запись в бд')
    print('2. Оценка прогностической модели')
    print('3. Построение графиков')
    menu = input()

    df1 = pd.read_csv("RMSI_0_0_817.csv")
    df1 = df1.rename(columns={"0.0": "RMSI_0_0", "2021.01.20 00:00:00": "Дата-Время"})
    df2 = pd.read_csv("RMSI_0_1_820.csv")
    df2 = df2.rename(columns={"0.0": "RMSI_0_1", "2021.01.20 00:00:00": "Дата-Время"})
    df3 = pd.read_csv("RMSI_0_2_822.csv")
    df3 = df3.rename(columns={"0.0": "RMSI_0_2", "2021.01.20 00:00:00": "Дата-Время"})
    df4 = pd.read_csv("RMSI_1_0_824.csv")
    df4 = df4.rename(columns={"0.0": "RMSI_1_0", "2021.01.20 00:00:00": "Дата-Время"})
    df5 = pd.read_csv("RMSV_2_0_829.csv")
    df5 = df5.rename(columns={"0.0": "RMSV_2_0", "2021.01.20 00:00:00": "Дата-Время"})
    df6 = pd.read_csv("RMSV_2_1_831.csv")
    df6 = df6.rename(columns={"0.0": "RMSV_2_1", "2021.01.20 00:00:00": "Дата-Время"})
    df7 = pd.read_csv("RMSV_2_2_833.csv")
    df7 = df7.rename(columns={"0.0": "RMSV_2_2", "2021.01.20 00:00:00": "Дата-Время"})
    df8 = pd.read_csv("RMSV_3_0_835.csv")
    df8 = df8.rename(columns={"0.0": "RMSV_3_0", "2021.01.20 00:00:00": "Дата-Время"})
    df1["Дата-Время"] = df1["Дата-Время"].apply(lambda x: pd.Timestamp(x))
    df1['Дата-Время'] = df1['Дата-Время'].astype('int64')
    dfsp = pd.concat(
        [df1, df2["RMSI_0_1"], df3["RMSI_0_2"], df4["RMSI_1_0"], df5["RMSV_2_0"], df6["RMSV_2_1"], df7["RMSV_2_2"],
         df8["RMSV_3_0"]], axis=1)

    dfsp['Дата-Время'] = pd.to_datetime(dfsp['Дата-Время'])
    dfsp.index = dfsp["Дата-Время"]
    dfsp = dfsp.drop("Дата-Время", axis=1)
    all_data = dfsp

    dfsp2 = pd.concat(
        [df1, df2["RMSI_0_1"], df3["RMSI_0_2"], df4["RMSI_1_0"], df5["RMSV_2_0"], df6["RMSV_2_1"], df7["RMSV_2_2"],
         df8["RMSV_3_0"]], axis=1)
    dfsp2 = dfsp2.loc[6500 < dfsp2.index]
    dfsp2 = dfsp2.loc[dfsp2.index < 11710]
    dfsp2 = dfsp2.loc[dfsp2['RMSI_0_1'] < 1]
    dfsp2 = dfsp2.set_index(np.arange(len(dfsp2.index)))
    dfsp2['Дата-Время'] = pd.to_datetime(dfsp2['Дата-Время'])
    dfsp2.index = dfsp2["Дата-Время"]
    dfsp2 = dfsp2.drop("Дата-Время", axis=1)

    if menu == '1':
        record(all_data, 'Electro')

    if menu == '2':

        fields = ['RMSI_0_0', 'RMSI_0_1', 'RMSI_0_2', 'RMSI_1_0', 'RMSV_2_0', 'RMSV_2_1', 'RMSV_2_2', 'RMSV_3_0']
        arr_lags = [793, 794, 794, 711, 850, 843, 637, 1]
        table = []
        for i in range(len(fields)):
            train = dfsp2[fields[i]][:len(dfsp2) - 100]
            train.index = pd.DatetimeIndex(train.index.values, freq=train.index.inferred_freq)

            test = dfsp2[fields[i]][len(dfsp2) - 100:]
            test.index = pd.DatetimeIndex(test.index.values, freq=test.index.inferred_freq)
            mod = AutoReg(train, lags=arr_lags[i])
            ar_model = mod.fit()
            pred = ar_model.predict(start=len(train), end=len(dfsp2)-1, dynamic=False)
            table.append(
                {'MSE': mean_squared_error(test, pred), 'RMSE': (mean_squared_error(test, pred))**(0.5),
                 'MAE': mean_absolute_error(test, pred), 'r2_score': r2_score(test, pred)})
            if i == 0:
                df = pd.DataFrame({'Дата-Время': pred.index})
            df[fields[i]] = pred.values
        dframe = pd.DataFrame(table, index=fields)
        df["Дата-Время"] = df["Дата-Время"].apply(lambda x: pd.Timestamp(x))
        df.index = df['Дата-Время']
        df = df.drop('Дата-Время', axis=1)
        print(dframe)
        print('Записать прогноз в бд?')
        if input() == 'yes':
            record(df, 'electroPred')

    if menu == '3':
        print('1. RMSI_0_0')
        print('2. RMSI_0_1')
        print('3. RMSI_0_2')
        print('4. RMSI_1_0')
        print('5. RMSV_2_0')
        print('6. RMSV_2_1')
        print('7. RMSV_2_2')
        print('8. RMSV_3_0')
        number_field = int(input('Выбрать поле '))
        field = 'RMSI_0_0'
        arr_lags = [793, 794, 794, 711, 850, 843, 637, 1]
        lag = arr_lags[0]
        fields = ['RMSI_0_0', 'RMSI_0_1', 'RMSI_0_2', 'RMSI_1_0', 'RMSV_2_0', 'RMSV_2_1', 'RMSV_2_2', 'RMSV_3_0']
        if 1 <= number_field <= 8:
            field = fields[number_field-1]
            lag = arr_lags[number_field-1]
        train = dfsp2[field][:len(dfsp2) - 100]
        train.index = pd.DatetimeIndex(train.index.values, freq=train.index.inferred_freq)
        test = dfsp2[field][len(dfsp2) - 100:]

        mod = AutoReg(train, lags=lag)
        ar_model = mod.fit()
        pred = ar_model.predict(start=len(train), end=len(dfsp2)-1, dynamic=False)
        pyplot.plot(pred, color='green')
        pyplot.plot(test, color='red')
        pyplot.show()

