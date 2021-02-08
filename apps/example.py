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
    st.title('We are going to show the application of ARIMA model in python using various libraries, including sktime, statsmodels, and pandas')
    st.write('Firstly, we need to import dependencies')

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
    register_matplotlib_converters()
            """
    st.code(code1, language="python")
    st.write("Secondly, let's download dataset and plot it")
    code2 = """
    y = load_airline()
    plot_series(y)
    """
    st.code(code2, language="python")
    y, df, rolling_mean, rolling_std = load_dataset()
    st.line_chart(df)
    st.write("Let's split data to train and test and plot it")
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

    st.write("Now we use pandas methods to calculate rolling mean and standard deviation, and make the corresponding plot")

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
    st.write("Then we use the augmented Dickey-Fuller test, and show the results.")
    with st.echo():
        result = adfuller(y)
        print('ADF Statistic: {}'.format(result[0]))
        print('p-value: {}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            print("\t{}: {}".format(key, value))

    st.text("Output:")
    output = st.empty()
    with st_capture(output.code):
        result = adfuller(y)
        print('ADF Statistic: {}'.format(result[0]))
        print('p-value: {}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            print("\t{}: {}".format(key, value))

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

    st.write("Precise function to plot the series with rolling mean and rolling standard deviation, which also shows the results of a Augmented Dickey-Fuller test")

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
    st.write("It is important to note that we adjust the data here before we use the function by substracting the mean")

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

    state.sync()
