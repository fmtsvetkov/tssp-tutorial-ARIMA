from chart_studio.plotly import iplot
from matplotlib.backends.backend_agg import RendererAgg
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
import plotly.express as px
import datetime
import chart_studio.plotly as py
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
from statsmodels.tsa.stattools import adfuller
from random import choices
from pandas.plotting import register_matplotlib_converters
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils.plotting import plot_series
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import r2_score
import streamlit as st
from io import StringIO
import sys
import pmdarima as pm
from contextlib import contextmanager, redirect_stdout
register_matplotlib_converters()
_lock = RendererAgg.lock


def try_read_df(f, index_col=None):
    try:
        return pd.read_csv(f, index_col=index_col)
    except:
        return pd.read_excel(f, index_col=index_col)


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


def adfuller_func(timeseries):
    result = adfuller(
        timeseries)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print("\t{}: {}".format(key, value))
    if result[0] > result[4]["5%"]:
        print(
            "Не удалось отклонить нулевую гипотезу - временной ряд нестационарный")
    else:
        print("Нулевая гипотеза отклонена – временной ряд стационарен")


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


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield


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
        # Example: state.value += 1
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


def reading_data_frame(file, index_col=None):
    df = try_read_df(file, index_col)
    return(df)


def app():
    """
    Part for creating web page
    """
    state = _get_state()

    st.title("ARIMA модель")
    state.uploaded_file = st.sidebar.file_uploader(
        "Выберите файл", type=['csv', 'xlsx', 'xls'])

    if state.uploaded_file is not None:
        state.data_frame = reading_data_frame(state.uploaded_file)
        state.column_to_predict = st.selectbox(
            'Выберите колонку для обучения', state.data_frame.columns, key="column_training")

        if state.column_to_predict is not None:

            # state.data_frame = state.data_frame.dropna(subset=[state.column_to_predict])

            state.tts = st.slider(
                'Выберите необходимую величину обучающей выборки',
                0.05, 0.5, (0.2), step=0.05, key='split_size'
            )
            state.y_train, state.y_test = temporal_train_test_split(
                state.data_frame[state.column_to_predict], test_size=state.tts)

            with _lock:
                fig0, ax0 = plot_series(
                    state.y_train, state.y_test, labels=['Train', 'Test'])
                st.pyplot(fig0)

            state.window = st.slider(
                'Выберите необходимую величину rolling window',
                1, 20, (5), step=1, key='window'
            )

            state.rolling_mean = state.data_frame[state.column_to_predict].rolling(
                window=state.window).mean()
            state.rolling_std = state.data_frame[state.column_to_predict].rolling(
                window=state.window).std()

            with _lock:
                fig1, ax1 = plt.subplots()
                state.data_frame[state.column_to_predict].plot()
                ax1.plot(state.rolling_mean, color='red',
                         label='Rolling Mean')
                ax1.plot(state.rolling_std, color='black',
                         label='Rolling Std')
                ax1.legend(loc='best')
                ax1.set_title('Rolling Mean & Standard Deviation')
                st.pyplot(fig1)

            # Dickey–Fuller test:
            with st_stdout("code"):
                adfuller_func(state.data_frame[state.column_to_predict])

    else:
        st.warning("Загрузите файл для начала работы")

    state.sync()
