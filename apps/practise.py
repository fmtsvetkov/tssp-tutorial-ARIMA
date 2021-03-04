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
    else:
        st.write("Загрузите файл для начала работы")

    state.sync()
