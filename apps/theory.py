from chart_studio.plotly import iplot
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
import plotly.express as px
import datetime
import chart_studio.plotly as py
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from random import choices
from string import ascii_letters
import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import r2_score
import streamlit as st

st.set_page_config(layout='wide')


def try_read_df(f):
    try:
        return pd.read_csv(f)
    except:
        return pd.read_excel(f)


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


def app():
    """
    Part for creating web page
    """
    state = _get_state()
    st.title("Теория ARIMA модели")
    st.markdown("Подход ARIMA к временным рядам заключается в том, что в первую очередь оценивается стационарность ряда. Различными тестами выявляются наличие единичных корней и порядок интегрированности временного ряда(обычно ограничиваются первым или вторым порядком). Далее при необходимости(если порядок интегрированности больше нуля) ряд преобразуется взятием разности соответствующего порядка и уже для преобразованной модели строится некоторая ARMA-модель, поскольку предполагается, что полученный процесс является стационарным, в отличие от исходного нестационарного процесса(разностно-стационарного или интегрированного процесса порядка d). Пусть задан временной ряд")
    st.latex(
        r'''\left(1-\sum_{i=1}^{p} \phi_{i} L^{i}\right) X_{t}=\left(1+\sum_{i=1}^{q} \theta_{i} L^{i}\right) \epsilon_{t}''')
    st.markdown("где t — целый индекс и $X_t$— вещественные числа. Тогда модель ARMA(p, q) задаётся следующем образом: где L — оператор задержки, $\phi_i$ — параметры авторегрессионной части модели, $\Theta_i$ — параметры скользящего среднего, а $\epsilon_t$ — значения ошибки. Обычно предполагают, что ошибки являются независимыми одинаково распределёнными случайными величинами из нормального распределения с нулевым средним. ARIMA(p, d, q) получается интегрированием ARMA(p, q).")

    st.latex(
        r'''\left(1-\sum_{i=1}^{p} \phi_{i} L^{i}\right)(1-L)^{d} X_{t}=\left(1+\sum_{i=1}^{q} \theta_{i} L^{i}\right) \epsilon_{t}''')
    st.markdown("где d — положительное целое, задающее уровень дифференцирования(если d=0, эта модель эквивалентна авторегрессионному скользящему среднему). И наоборот, применяя почленное дифференцирование d раз к модели ARMA(p, q), получим модель ARIMA(p, d, q). Заметим, что дифференцировать надо только авторегрессионную часть. Важно отметить, что не все сочетания параметров дают «хорошую» модель. В частности, чтобы получить стационарную модель требуется выполнение некоторых условий. Существует несколько известных частных случаев модели ARIMA. Например, ARIMA(0, 1, 0), задающая")
    st.latex(r'''X_{t}=X_{t-1}+\epsilon_{t}''')
    st.markdown("является моделью случайных блужданий")

    state.sync()
