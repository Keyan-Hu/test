import pandas as pd
import numpy as np
from numpy.fft import fft
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense, TimeDistributed
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from functools import partial


## 数据预处理
def preprocess_data(csv_path, variable, start_date=None, end_date=None):
    # 读取CSV文件
    data = pd.read_csv(csv_path)

    # 提取日期和指定变量列到temp_data
    temp_data = data[['date', variable]]

    # 创建日期索引
    temp_data['date'] = pd.to_datetime(temp_data['date'])
    temp_data.set_index('date', inplace=True)

    # 根据起始和结束时间筛选数据
    if start_date is not None:
        temp_data = temp_data.loc[temp_data.index >= start_date]
    if end_date is not None:
        temp_data = temp_data.loc[temp_data.index <= end_date]

    return temp_data


## 数据归一化
def normalize_data(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data, scaler

## 线性回归方法
def linear_regression_methods(data, models):
    # 准备数据
    x = np.array(range(len(data))).reshape(-1, 1)
    y = data[variable].values.reshape(-1, 1)

    # 训练模型
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'ElasticNet Regression': ElasticNet(alpha=0.1, l1_ratio=0.5)}
    trained_models = {}
    for name, model in models.items():
        model.fit(x, y)
        data[name] = model.predict(x)
        trained_models[name] = model
    return data,trained_models

# 输出具体拟合函数的方程式
def print_linear_regression_equation(trained_models):
    for name, model in trained_models.items():
        slope = model.coef_[0].item()
        intercept = model.intercept_.item()
        print(f"线性回归方程 ({name}): y = {slope:.2f}x + {intercept:.2f}")

## Harmonic谐波回归
def harmonic_model(x, *params):
    """
    返回给定参数的谐波函数在给定输入处的值。
    Args:
        x: 输入，函数值将在该输入处计算。
        *params: 可变长度参数列表，包含每个谐波的振幅和相位。
    Returns:
        给定输入处的谐波函数值。
    """
    result = params[0] + params[1] * x
    omega_2pi = params[2]  # 新的ω参数 (2πω)
    for i in range(3, len(params), 2):
        if i + 1 < len(params):
            result += params[i] * np.cos(omega_2pi * ((i - 3) // 2 + 1) * x) + params[i + 1] * np.sin(omega_2pi * ((i - 3) // 2 + 1) * x)
    return result

def harmonic_regression(data, num_harmonics=1, optimize_initial_guess=False):
    """
    对给定数据进行谐波回归。
    Args:
        data: 要进行回归的数据。
        num_harmonics: 要使用的谐波级数。1阶表示线性表示，2阶表示线性和harmonic表示，以此类推。
        optimize_initial_guess: 是否优化初始参数估计。
    Returns:
        包含谐波回归结果的数据，以及每个谐波的振幅和相位参数。
    """
    x = np.array(range(len(data)))
    y = data[variable].values

    if optimize_initial_guess:
        # 预处理数据
        y_smooth = savgol_filter(y, window_length=51, polyorder=3)

        # 使用傅里叶变换找到主要频率分量
        y_fft = fft(y_smooth)
        frequencies = np.fft.fftfreq(len(y_smooth))
        max_freq_index = np.argmax(np.abs(y_fft[1:len(y_fft)//2]))
        main_frequency = frequencies[max_freq_index + 1]

        # 设置初始参数估计
        initial_guess = [np.mean(y), 0] + [0, 0] * num_harmonics
        initial_guess[2] = main_frequency
        print(f"Harmonic谐波参数(阶次{len(initial_guess)//2-1})已自动寻优")
    else:
        print("Harmonic谐波参数自动寻优未启动")
        np.random.seed(1)
        initial_guess = np.random.rand(2 + 2 * num_harmonics) * 10

    params, _ = curve_fit(harmonic_model, x, y, p0=initial_guess, maxfev=50000)

    if optimize_initial_guess:
        best_params = None
        best_residual = float('inf')

        for _ in range(10):
            try:
                params, _ = curve_fit(harmonic_model, x, y, p0=initial_guess, maxfev=50000)
                y_harmonic = harmonic_model(x, *params)
                residual = np.sum((y - y_harmonic)**2)

                if residual < best_residual:
                    best_residual = residual
                    best_params = params
            except RuntimeError:
                pass

        params = best_params

    y_harmonic = harmonic_model(x, *params)
    data['Harmonic Regression'] = y_harmonic

    return data, params

def print_harmonic_regression_equation(params):
    """
    打印给定谐波回归的方程。
    Args:
        params: 包含谐波回归振幅和相位参数的列表。
    """
    omega_2pi = params[2]  # 新的ω参数 (2πω)
    equation = f"谐波回归方程: y = {params[0]:.2f} + {params[1]:.2f}t"
    for i in range(3, len(params), 2):
        if i + 1 < len(params):
            equation += f" + {params[i]:.2f} * cos({omega_2pi:.2f} * {((i - 3) // 2 + 1)}t) + {params[i + 1]:.2f} * sin({omega_2pi:.2f} * {((i - 3) // 2 + 1)}t)"
    print(equation)

def find_best_num_harmonics(data, max_harmonics=10, cv=5):
    """
    寻找谐波回归中最佳的谐波数量。
    Args:
        data: 要进行回归的数据。
        max_harmonics: 最大谐波数量。
        cv: 交叉验证的折数。
    Returns:
        最佳谐波数量。
    """
    x = np.array(range(len(data)))
    y = data[variable].values

    best_harmonics = 0
    best_score = float('-inf')

    for num_harmonics in range(1, max_harmonics + 1):
        initial_guess = [0] * (3 + 2 * num_harmonics)
        params, _ = curve_fit(harmonic_model, x, y, p0=initial_guess, maxfev=50000)

        y_harmonic = harmonic_model(x, *params)
        score = np.sqrt(np.mean((y - y_harmonic) ** 2))  # 使用RMSE作为评分标准

        if score > best_score:
            best_score = score
            best_harmonics = num_harmonics

    return best_harmonics

## Savitzky-Golay滤波拟合
def savitzky_golay_filter(data, window_length=20, polyorder=3):
    if window_length >= len(data):
        window_length = len(data) - 1 if len(data) % 2 == 0 else len(data)
    return savgol_filter(data, window_length, polyorder)

# 寻找最优的Savitzky-Golay滤波器参数
def find_best_sg_params(data, min_window_length=5, max_window_length=None, fixed_polyorder=None, max_polyorder=5, n_splits=5):
    """
    找到一组信号处理参数，以对给定数据进行平滑处理。

    参数：
    data：要进行平滑处理的数据。
    min_window_length：窗口长度的最小值。
    max_window_length：窗口长度的最大值。None：则使用数据长度作为最大值。
    fixed_polyorder：多项式阶数。如果为None，则自动寻优。
    max_polyorder：多项式阶数的最大值。
    n_splits：要进行交叉验证的次数。

    返回：
    一个元组，包含最优的窗口长度和阶数。
    """
    # 处理最大窗口长度的边界条件
    if max_window_length is None:
        max_window_length = len(data) // 2
    max_window_length = min(max_window_length, len(data) - 1)
    if max_window_length % 2 == 0:
        max_window_length -= 1
    if min_window_length % 2 == 0:
        max_window_length -= 1

    # 创建TimeSeriesSplit类的实例
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # 初始化最优参数
    best_mse = np.inf
    best_window_length = min_window_length
    best_polyorder = 1
    
    # 如果fixed_polyorder为None，遍历所有阶数寻找最优参数
    if fixed_polyorder is None:
        for polyorder in range(1, max_polyorder + 1):
            for window_length in range(min_window_length, max_window_length + 1, 2):
                # 添加这个条件以确保 polyorder 小于 window_length
                if polyorder >= window_length:
                    continue
                mse = 0
                for train_index, test_index in tscv.split(data):
                    train_data, test_data = data[train_index], data[test_index]
                    # 对训练数据进行滤波
                    filtered_train_data = savitzky_golay_filter(train_data, window_length=window_length, polyorder=polyorder)
                    # 计算均方误差
                    mse += mean_squared_error(test_data, filtered_train_data[:len(test_data)])
                # 计算平均均方误差
                mse /= n_splits
                # 更新最优参数
                if mse < best_mse:
                    best_mse = mse
                    best_window_length = window_length
                    best_polyorder = polyorder
        print(f"Savitzky-Golay滤波器参数(窗口长度{best_window_length}+阶次{best_polyorder})已自动寻优")
    # 如果fixed_polyorder为整数，使用固定的阶数寻找最优参数
    else:
        print("Savitzky-Golay滤波器参数(窗口长度)已自动寻优")
        best_polyorder = fixed_polyorder
        for window_length in range(min_window_length, max_window_length + 1, 2):
            # 确保 polyorder 小于 window_length
            if best_polyorder >= window_length:
                continue
            mse = 0
            for train_index, test_index in tscv.split(data):
                train_data, test_data = data[train_index], data[test_index]
                # 对训练数据进行滤波
                filtered_train_data = savitzky_golay_filter(train_data, window_length=window_length, polyorder=best_polyorder)
                # 计算均方误差
                mse += mean_squared_error(test_data, filtered_train_data[:len(test_data)])
            # 计算平均均方误差
            mse /= n_splits
            # 更新最优参数
            if mse < best_mse:
                best_mse = mse
                best_window_length = window_length

    return best_window_length, best_polyorder

def find_best_polyorder(data, min_polyorder=2, max_polyorder=10, window_length=20, n_splits=5):
    """
    根据均方误差选择最优的Savitzky-Golay滤波器多项式阶数
    :param data: 待处理的数据
    :param min_polyorder: 最小的多项式阶数，默认为2
    :param max_polyorder: 最大的多项式阶数，默认为10
    :param window_length: 滤波器窗口长度，默认为20
    :param n_splits: 交叉验证的折数，默认为5
    :return: 最优的多项式阶数
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)  # 创建时间序列交叉验证器
    best_mse = np.inf  # 初始化最小的均方误差为正无穷大
    best_polyorder = min_polyorder  # 初始化最优的多项式阶数为最小的多项式阶数

    for polyorder in range(min_polyorder, max_polyorder + 1):
        # 遍历多项式阶数的范围（从最小到最大），计算均方误差
        mse = 0
        for train_index, test_index in tscv.split(data):
            train_data, test_data = data[train_index], data[test_index]
            filtered_train_data = savitzky_golay_filter(train_data, window_length=window_length, polyorder=polyorder)
            mse += mean_squared_error(test_data, filtered_train_data[:len(test_data)])
        mse /= n_splits  # 计算平均的均方误差
        if mse < best_mse:  # 如果当前的均方误差小于最小的均方误差，则更新最优的多项式阶数
            best_mse = mse
            best_polyorder = polyorder

    return best_polyorder  # 返回最优的多项式阶数


## RNN拟合
def rnn_fit0(data, epochs=500, batch_size=10, n_units=50, learning_rate=0.001):
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data[variable].values.reshape(-1, 1)
    
    # Normalize input data
    X, X_scaler = normalize_data(X)
    y, y_scaler = normalize_data(y)

    # 准备RNN数据
    X_rnn = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    y_rnn = np.reshape(y, (y.shape[0], y.shape[1]))

    # RNN拟合
    model_rnn = Sequential()
    model_rnn.add(SimpleRNN(units=n_units, input_shape=(X_rnn.shape[1], X_rnn.shape[2]), return_sequences=True))
    model_rnn.add(SimpleRNN(units=n_units))
    model_rnn.add(Dense(1))
    model_rnn.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
    model_rnn.fit(X_rnn, y_rnn, epochs=epochs, batch_size=batch_size, verbose=0)
    
    data['RNN'] = y_scaler.inverse_transform(model_rnn.predict(X_rnn))
    return data

# RNN--参数调整后
def rnn_fit1(data, epochs=300, batch_size=10, neurons=64, layers=3, learning_rate=0.001):
    from keras.optimizers import Adam

    X = np.array(range(len(data))).reshape(-1, 1)
    y = data[variable].values.reshape(-1, 1)
    
    X_rnn = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    y_rnn = np.reshape(y, (y.shape[0], y.shape[1]))

    model_rnn = Sequential()
    model_rnn.add(SimpleRNN(units=neurons, return_sequences=True, input_shape=(X_rnn.shape[1], X_rnn.shape[2])))
    
    for _ in range(layers - 2):
        model_rnn.add(SimpleRNN(units=neurons, return_sequences=True))

    model_rnn.add(SimpleRNN(units=neurons))
    model_rnn.add(Dense(1))

    model_rnn.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
    model_rnn.fit(X_rnn, y_rnn, epochs=epochs, batch_size=batch_size, verbose=0)
    
    data['RNN'] = model_rnn.predict(X_rnn)
    return data

## LSTM拟合
def lstm_fit0(data, epochs=500, batch_size=10, n_units=50, learning_rate=0.001):
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data[variable].values.reshape(-1, 1)
    
    # Normalize input data
    X, X_scaler = normalize_data(X)
    y, y_scaler = normalize_data(y)

    # 准备LSTM数据
    X_lstm = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    y_lstm = np.reshape(y, (y.shape[0], y.shape[1]))

    # LSTM拟合
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=n_units, input_shape=(X_lstm.shape[1], X_lstm.shape[2]), return_sequences=True))
    model_lstm.add(LSTM(units=n_units))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
    model_lstm.fit(X_lstm, y_lstm, epochs=epochs, batch_size=batch_size, verbose=0)

    data['LSTM'] = y_scaler.inverse_transform(model_lstm.predict(X_lstm))
    return data

# LSTM参数调整后
def lstm_fit1(data, epochs=300, batch_size=10, neurons=64, layers=3, learning_rate=0.001):
    from keras.optimizers import Adam

    X = np.array(range(len(data))).reshape(-1, 1)
    y = data[variable].values.reshape(-1, 1)
    
    X_lstm = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    y_lstm = np.reshape(y, (y.shape[0], y.shape[1]))

    model_lstm = Sequential()
    model_lstm.add(LSTM(units=neurons, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
    
    for _ in range(layers - 2):
        model_lstm.add(LSTM(units=neurons, return_sequences=True))

    model_lstm.add(LSTM(units=neurons))
    model_lstm.add(Dense(1))

    model_lstm.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
    model_lstm.fit(X_lstm, y_lstm, epochs=epochs, batch_size=batch_size, verbose=0)

    data['LSTM'] = model_lstm.predict(X_lstm)

    return data

##分别提取数据的年与月平均值
def monthly_mean(data):

    monthly_data = data.resample('M').mean()
    # 删除首月不满一月的数据
    if data.index[0].day != 1:
        monthly_data = monthly_data.iloc[1:]
    # 删除尾月不满一月的数据
    if data.index[-1].day != data.index[-1].days_in_month:
        monthly_data = monthly_data.iloc[:-1]

    return monthly_data

def yearly_mean(data):
    yearly_data = data.resample('Y').mean()
    # 删除首年不满一年的数据
    if data.index[0].day_of_year != 1:
        yearly_data = yearly_data.iloc[1:]
    # 删除尾年不满一年的数据
    days_in_year = 366 if data.index[-1].is_leap_year else 365
    if data.index[-1].day_of_year != days_in_year:
        yearly_data = yearly_data.iloc[:-1]
    
    return yearly_data

## 绘制子图
def plot_subplots(data, models):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True)
    axes = axes.flatten()
    # 原始数据
    axes[0].plot(data.index, data[variable], label='Original Data', alpha=0.5)
    axes[0].set_title('Original Data')

    # 线性回归方法
    color_map = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2']
    for i, (name, color) in enumerate(zip(models.keys(), color_map)):
        axes[1].plot(data.index, data[name], label=name, color=color)
    axes[1].plot(data.index, data[variable], label='Original Data', alpha=0.5)
    axes[1].set_title('Linear Regression Methods')

    # Savitzky-Golay滤波拟合
    axes[2].plot(data.index, data[variable], label='Original Data', alpha=0.5)
    axes[2].plot(data.index, data['Savitzky-Golay'], label='Savitzky-Golay', color='#8ECFC9')
    axes[2].set_title('Savitzky-Golay')
    # 谐波回归
    axes[3].plot(data.index, data[variable], label='Original Data', alpha=0.5)
    axes[3].plot(data.index, data['Harmonic Regression'], label='Harmonic Regression', color='#BEB8DC')
    axes[3].set_title('Harmonic Regression')
    # RNN拟合
    axes[4].plot(data.index, data[variable], label='Original Data', alpha=0.5)
    axes[4].plot(data.index, data['RNN'], label='RNN', color='#FFBE7A')
    #axes[4].plot(data.index, MinMaxScaler().fit_transform(data[variable].values.reshape(-1, 1)), label='Normalized Data', alpha=0.3)
    axes[4].set_title('RNN')

    # LSTM拟合
    axes[5].plot(data.index, data[variable], label='Original Data', alpha=0.5)
    axes[5].plot(data.index, data['LSTM'], label='LSTM', color='#FA7F6F')
    #axes[5].plot(data.index, MinMaxScaler().fit_transform(data[variable].values.reshape(-1, 1)), label='Normalized Data', alpha=0.3)
    axes[5].set_title('LSTM')

    # 设置子图属性
    for ax in axes:
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel(variable)
        ax.set_ylim(0, 60)  # 添加这一行来设置纵轴范围为0-100

    # 调整子图间距
    fig.tight_layout()

    plt.show()

## 数据处理子函数，参数为data、day、month、year
def Model_frequency(data,models,parameter='year'):
    """
    对输入数据进行预处理
    :param data: 待处理的数据
    :param models: 使用的模型列表
    :param parameter: 预处理参数，默认为'year',有'day','month','year'三种选择
    :return: 处理后的数据
    """
    cilp_data = data
    if parameter == 'month':
        cilp_data = monthly_mean(data)
    if parameter == 'year':
        cilp_data = yearly_mean(data)
    #拟合模型
    cilp_data,linear_regression_models = linear_regression_methods(cilp_data, models)
    #best_num_harmonics = find_best_num_harmonics(cilp_data)
    cilp_data, harmonic_regression_params = harmonic_regression(cilp_data, 2, True)
    best_window_length, best_polyorder = find_best_sg_params(cilp_data[variable].values, fixed_polyorder=None)  # fixed_polyorder=3 表示固定阶数为 3
    cilp_data['Savitzky-Golay'] = savitzky_golay_filter(cilp_data[variable], best_window_length, best_polyorder)
    cilp_data = rnn_fit0(cilp_data) # 调整了epochs和batch_size
    cilp_data = lstm_fit0(cilp_data) # 调整了epochs和batch_size

    return cilp_data,linear_regression_models,harmonic_regression_params

## RMSE计算子函数
def calculate_rmse(data):
    """
    计算RMSE指标
    
    参数：
    data: 包含真实值和预测值
    返回值：
    RMSE值
    """
    
    rmse_values = {}
    true_col = data.columns[0]
    for col in data.columns[1:]:
        if  col != 'date':
            rmse_values[col] = np.sqrt(mean_squared_error(data[true_col], data[col]))
    return pd.Series(rmse_values)



## 主函数
#定义变量列表和当前变量
variables = ['volumetric_soil_water_layer_1',
             'volumetric_soil_water_layer_2',
             'volumetric_soil_water_layer_3',
             'volumetric_soil_water_layer_4',
             'sub_surface_runoff_sum',
             'surface_runoff_sum',
             'soil_temperature_level_1',
             'soil_temperature_level_2',
             'soil_temperature_level_3',
             'soil_temperature_level_4',
             'dewpoint_temperature_2m',
             'temperature_2m',
             'skin_temperature']

variable = 'volumetric_soil_water_layer_1'

#读取并处理数据
data = preprocess_data(r"C:\Users\39050\Desktop\2023\Analysis of soil moisture in Qinghai-Tibet Plateau based on long time series\A. GEE_Data_gather\A.GEE_QZ_data\Specified_area_QZ\experimental_area2\Daily\ERA5-daily-mult_variables.csv", variable) 

#设置起始时间和结束时间
'''
start_date = '1967-07-11'
end_date = '2023-1-1'
'''
start_date = '2010-01-1'
end_date = '2018-1-1'
#截取指定时间范围的数据
data = data[start_date:end_date]

#定义模型字典
models = {
'Linear Regression': LinearRegression(),
'Ridge Regression': Ridge(alpha=1.0),
'Lasso Regression': Lasso(alpha=0.1),
'ElasticNet Regression': ElasticNet(alpha=0.1, l1_ratio=0.5)}

#设置模型频率并进行数据处理
reslut_data,linear_regression_models,harmonic_regression_params = Model_frequency(data,models,parameter='day')

#计算RMSE
rmse_values = calculate_rmse(reslut_data)
print("各列预测值的RMSE值如下：")
for col, rmse in rmse_values.items():
    print(f"{col}的RMSE值为{rmse:.4f}")

#输出函数具体表达式
print_linear_regression_equation(linear_regression_models)
print_harmonic_regression_equation(harmonic_regression_params)

#绘制子图
plot_subplots(reslut_data, models)