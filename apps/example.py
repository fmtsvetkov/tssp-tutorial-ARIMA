from matplotlib.backends.backend_agg import RendererAgg
from streamlit.source_util import open_python_file
import matplotlib.pyplot as plt
import base64
import plotly.express as px
import altair as alt
import datetime
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
from pandas.plotting import register_matplotlib_converters
from sktime.forecasting.model_selection import temporal_train_test_split
from contextlib import contextmanager, redirect_stdout
from io import StringIO
import sys
import pmdarima as pm
register_matplotlib_converters()
_lock = RendererAgg.lock


def try_read_df(f):
    try:
        return pd.read_csv(f)
    except:
        return pd.read_excel(f)


@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=True)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return(href)


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(
            self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


def get_stationarity(timeseries):

    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    # rolling statistics plot
    original = timeseries.plot()
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Dickey–Fuller test:
    result = adfuller(timeseries)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


@st.cache
def load_dataset():
    y = load_airline()
    df = pd.DataFrame(y)
    df.index = df.index.to_timestamp()
    rolling_mean = df.rolling(window=12).mean()
    rolling_std = df.rolling(window=12).std()
    return(y, df, rolling_mean, rolling_std)


def plots(df, rolling_mean, rolling_std, data_label):
    fig, ax = plt.subplots()
    ax.plot(df, label=data_label)
    ax.plot(rolling_mean, color='red', label='Rolling Mean')
    ax.plot(rolling_std, color='black', label='Rolling Std')
    ax.legend(loc='best')
    ax.set_title('Rolling Mean & Rolling Standard Deviation')
    return(fig, ax)


def app():
    """
    Part for creating web page
    """
    state = _get_state()
    st.title('Туториал по исследованию временных рядов при помощи Python.')
    st.write('Мы рассмотрим как работать с ARIMA-моделью в Python. Есть несколько различных библиотек, которые позволяют исследовать временные ряды. Мы решили использовать sktime, statsmodels, и классическую библиотеку pandas. Где это будет возможно, мы даже сравним возможности этих библиотек, хотя в основном наш выбор пал на самие понятные и негромоздкие решения.')

    code1 = """
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima_model import ARIMA
    from sktime.datasets import load_airline
    from sktime.utils.plotting import plot_series
    from pandas.plotting import register_matplotlib_converters
    from sktime.forecasting.model_selection import temporal_train_test_split
    import pmdarima as pm
    register_matplotlib_converters()
            """
    st.code(code1, language="python")
    st.write("Для начала, рассмотрим самые стандартные операции с времеными рядами. У нас есть шаблонный набор даных об авиаперелётах, собранный по конкретонй компании. В следующих нескольких строках мы просто загружаем эти самые данные, и что-нибудь делаем. Если конкретней, строим график, делаем train-test split. Это, в целом, стандартный подход к временным рядам, но от этого не менее важный.")
    code2 = """
    y = load_airline()
    plot_series(y)
    """
    st.code(code2, language="python")
    y, df, rolling_mean, rolling_std = load_dataset()
    st.line_chart(df)
    st.write("На получившийся график можно взглянуть повнимательней. Сразу видна сезонность данных, явные пики и проседания, Ну и прииерно на глазок можно углядеть тренд на увеличение среднегодового колличества пассажиров, даже несмотря на сезонные провалы.")
    code3 = """
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    plot_series(y_train, y_test, labels=['Train', 'Test'])
    """
    st.code(code3, language="python")
    y_train, y_test = temporal_train_test_split(
        y, test_size=24)

    with _lock:
        fig0, ax0 = plot_series(
            y_train, y_test, labels=['Train', 'Test'])
        st.pyplot(fig0)

    st.write("А вот тут мы уже сделали разбивку данных на тестовый и обучающий набор. Синим обозначен обучающий, а золотым - тестовый.")

    st.write("Посмотрим на график. Синим показано наблюдаемое количество пассажиров авиалиний. Красным показано скользящее среднее, а чёрным - скользящее стандартное отклонение.")
    code4 = """
    df = pd.DataFrame(y)
    df.index = df.index.to_timestamp()
    rolling_mean = df.rolling(window = 12).mean()
    rolling_std = df.rolling(window = 12).std()
    df.plot()
    plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
    plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Rolling Standard Deviation')
    plt.show()
    """
    st.code(code4, language="python")

    with _lock:
        fig1, ax1 = plots(df, rolling_mean,
                          rolling_std, 'Number of airline passengers')
        """
        fig1, ax1 = plt.subplots()
        ax1.plot(df, label='Number of airline passengers')
        ax1.plot(rolling_mean, color='red', label='Rolling Mean')
        ax1.plot(rolling_std, color='black', label='Rolling Std')
        ax1.legend(loc='best')
        ax1.set_title('Rolling Mean & Rolling Standard Deviation')
        """
        st.pyplot(fig1)

    y = y
    st.write("Здесь мы переходим к немного более сложным вещам. Мы применяем к нашему ряду расширенный тест Дики - Фуллера.")
    with st.echo():
        result = adfuller(y)
        print('ADF Statistic: {}'.format(result[0]))
        print('p-value: {}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            print("\t{}: {}".format(key, value))
        if result[0] > result[4]["5%"]:
            print("Не удалось отклонить нулевую гипотезу - временной ряд нестационарный")
        else:
            print("Нулевая гипотеза отклонена – временной ряд стационарен")

    st.text("Output:")
    output = st.empty()
    with st_capture(output.code):
        result = adfuller(y)
        print('ADF Statistic: {}'.format(result[0]))
        print('p-value: {}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            print("\t{}: {}".format(key, value))
        if result[0] > result[4]["5%"]:
            print("Не удалось отклонить нулевую гипотезу - временной ряд нестационарный")
        else:
            print("Нулевая гипотеза отклонена – временной ряд стационарен")

    code5 = """
    df_log = np.log(y)
    df_log.plot()
    """
    st.code(code5, language="python")

    df_log = np.log(y)
    df_log.index = df_log.index.to_timestamp()

    with _lock:
        fig2, ax2 = plt.subplots()
        ax2.plot(df_log, label='Log')
        ax2.legend(loc='best')
        ax2.set_title('Logged number of airline passengers')
        st.pyplot(fig2)

    st.write("Далее будет функция, считающая наши скользящие средние, делающая тест Дики - Фуллера, и заодно показывающая график.")

    with st.echo():
        def get_stationarity(timeseries):

            # rolling statistics
            rolling_mean = timeseries.rolling(window=12).mean()
            rolling_std = timeseries.rolling(window=12).std()

            # rolling statistics plot
            timeseries.plot()
            plt.plot(rolling_mean, color='red', label='Rolling Mean')
            plt.plot(rolling_std, color='black', label='Rolling Std')
            plt.legend(loc='best')
            plt.title('Rolling Mean & Standard Deviation')
            plt.show(block=False)

            # Dickey–Fuller test:
            result = adfuller(timeseries)
            print('ADF Statistic: {}'.format(result[0]))
            print('p-value: {}'.format(result[1]))
            print('Critical Values:')
            for key, value in result[4].items():
                print('\t{}: {}'.format(key, value))
            if result[0] > result[4]["5%"]:
                print(
                    "Не удалось отклонить нулевую гипотезу - временной ряд нестационарный")
            else:
                print("Нулевая гипотеза отклонена – временной ряд стационарен")

    st.write("Отметим, что тут мы немного корректируем данные, вычитая среднее.")

    code6 = """
    rolling_mean = df_log.rolling(window=12).mean()
    df_log_minus_mean = df_log - rolling_mean
    df_log_minus_mean.dropna(inplace=True)
    get_stationarity(df_log_minus_mean)
    """
    st.code(code6, language="python")
    rolling_mean = df_log.rolling(window=12).mean()
    df_log_minus_mean = df_log - rolling_mean
    df_log_minus_mean.dropna(inplace=True)
    rolling_mean_log_minus_mean = df_log_minus_mean.rolling(
        window=12).mean()
    rolling_std_log_minus_mean = df_log_minus_mean.rolling(
        window=12).std()

    with _lock:
        fig3, ax3 = plots(df_log_minus_mean, rolling_mean_log_minus_mean,
                          rolling_std_log_minus_mean, 'Number of airline passengers (log minus mean)')
        """
        fig3, ax3 = plt.subplots()
        ax3.plot(df_log, label='Log')
        ax3.legend(loc='best')
        ax3.set_title('Logged number of airline passengers')
        """
        st.pyplot(fig3)
    output = st.empty()
    with st_capture(output.code):
        result = adfuller(df_log_minus_mean)
        print('ADF Statistic: {}'.format(result[0]))
        print('p-value: {}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            print("\t{}: {}".format(key, value))
        if result[0] > result[4]["5%"]:
            print("Не удалось отклонить нулевую гипотезу - временной ряд нестационарный")
        else:
            print("Нулевая гипотеза отклонена – временной ряд стационарен")
    st.write("График сверху показывает получившийся результат. Поверх количества пассажиров можно увидеть скользящие соеднее и стандартное отклонение.")
    code7 = """
    rolling_mean_exp_decay = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
    df_log_exp_decay = df_log - rolling_mean_exp_decay
    df_log_exp_decay.dropna(inplace=True)
    get_stationarity(df_log_exp_decay)
    """
    st.code(code7, language="python")
    rolling_mean_exp_decay = df_log.ewm(
        halflife=12, min_periods=0, adjust=True).mean()

    df_log_exp_decay = df_log - rolling_mean_exp_decay
    df_log_exp_decay.dropna(inplace=True)
    rolling_mean_log_exp_decay = df_log_exp_decay.rolling(
        window=12).mean()
    rolling_std_log_exp_decay = df_log_exp_decay.rolling(
        window=12).std()
    with _lock:
        fig4, ax4 = plots(df_log_exp_decay, rolling_mean_log_exp_decay,
                          rolling_std_log_exp_decay, 'Number of airline passengers')
        st.pyplot(fig4)
    output1 = st.empty()
    with st_capture(output1.code):
        result = adfuller(df_log_exp_decay)
        print('ADF Statistic: {}'.format(result[0]))
        print('p-value: {}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            print("\t{}: {}".format(key, value))
        if result[0] > result[4]["5%"]:
            print("Не удалось отклонить нулевую гипотезу - временной ряд нестационарный")
        else:
            print("Нулевая гипотеза отклонена – временной ряд стационарен")
    code8 = """
    df_log_shift = df_log - df_log.shift()
    df_log_shift.dropna(inplace=True)
    get_stationarity(df_log_shift)
    """
    st.code(code8, language="python")
    df_log_shift = df_log - df_log.shift()

    df_log_shift.dropna(inplace=True)
    rolling_mean_shift = df_log_shift.rolling(
        window=12).mean()
    rolling_std_shift = df_log_shift.rolling(
        window=12).std()
    with _lock:
        fig5, ax5 = plots(df_log_shift, rolling_mean_shift,
                          rolling_std_shift, 'Number of airline passengers')
        st.pyplot(fig5)
    output2 = st.empty()
    with st_capture(output2.code):
        result = adfuller(df_log_shift)
        print('ADF Statistic: {}'.format(result[0]))
        print('p-value: {}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            print("\t{}: {}".format(key, value))
        if result[0] > result[4]["5%"]:
            print("Не удалось отклонить нулевую гипотезу - временной ряд нестационарный")
        else:
            print("Нулевая гипотеза отклонена – временной ряд стационарен")
    st.write("Мы рассмотрели как можно изменять датасет для того, чтобы получить стационарный временной ряд. Теперь мы применим autoarima, для автоматического расчета оптимальных параметров модели")

    model = pm.auto_arima(y_train, seasonal=True, m=12)

    # make your forecasts
    # predict N steps into the future
    forecasts = model.predict(y_test.shape[0])

    x = np.arange(y.shape[0])
    code9 = """
model = pm.auto_arima(y_train, seasonal=True, m=12)

forecasts = model.predict(y_test.shape[0])

x = np.arange(y.shape[0])
plt.plot(x[:120], y_train, c='blue')
plt.plot(x[120:], forecasts, c='green')
plt.show()
    """
    st.code(code9, language="python")

    st.write("Приведенный код обучает модель с автоподбором параметров на тренировочных данных (y_train), строит прогноз на количество элементов в тестовых данных, а затем мы строим график прогнозных значений")
    fig6, ax6 = plt.subplots()
    ax6.plot(x[:120], y_train, c='blue', label='Тренировочные данные')
    ax6.plot(x[120:], forecasts, c='green', label='Предсказанные значения')
    ax6.legend(loc='best')
    st.pyplot(fig6)

    st.write('Теперь посмотрим какие параметры у нашей модели')
    code10 = """
model.summary()
    """
    st.code(code10, language="python")
    output3 = st.empty()
    with st_capture(output3.code):
        print(model.summary())

    code11 = """
# Create predictions for the future, evaluate on test
preds, conf_int = model.predict(
    n_periods=y_test.shape[0], return_conf_int=True)

# #############################################################################
# Plot the points and the forecasts

df.index = df.index.to_timestamp()

plt.plot(df.index[:y_train.shape[0]], y_train, alpha=0.75)
plt.plot(df.index[y_train.shape[0]:], preds, alpha=0.75)  # Forecasts
plt.scatter(df.index[y_train.shape[0]:], y_test,
            alpha=0.4, marker='x')  # Test data
plt.fill_between(df.index[-preds.shape[0]:],
                 conf_int[:, 0], conf_int[:, 1],
                 alpha=0.1, color='b')
    """
    st.code(code11, language="python")
    st.write(
        "Теперь мы построим график предсказаний, а Х будут обозначать реальные данные")
    # Create predictions for the future, evaluate on test
    preds, conf_int = model.predict(
        n_periods=y_test.shape[0], return_conf_int=True)

    # #############################################################################
    # Plot the points and the forecasts

    fig7, ax7 = plt.subplots()
    ax7.plot(df.index[:y_train.shape[0]], y_train, alpha=0.75)
    ax7.plot(df.index[y_train.shape[0]:], preds, alpha=0.75)  # Forecasts
    ax7.scatter(df.index[y_train.shape[0]:], y_test,
                alpha=0.4, marker='x')  # Test data
    ax7.fill_between(df.index[-preds.shape[0]:],
                     conf_int[:, 0], conf_int[:, 1],
                     alpha=0.1, color='b')
    st.pyplot(fig7)
    st.write(
        "Как мы видим, модель достаточно точно описала график, давайте теперь оценим нашу модель метрикой MAPE (средняя абсолютная ошибка в процентах)")
    code12 = """
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(mean_absolute_percentage_error(y_test, preds))
    """
    st.code(code12, language="python")
    output4 = st.empty()
    with st_capture(output4.code):
        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        print(mean_absolute_percentage_error(y_test, preds))
    st.write("10% отклонение выглядит отлично, модель справилась на ура")
    state.sync()
