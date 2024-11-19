# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import inspect
import textwrap
import pandas as pd
import altair as alt
from utils import show_code
import xml.etree.ElementTree as ET
#import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from urllib.error import URLError


def health_kit_demo():
    #@st.experimental_memo
    @st.cache_data
    def get_data():
    #def get_df_xml(filename):
        # create element tree object 
        tree = ET.parse('export.xml') 
        # for every health record, extract the attributes into a dictionary (columns). Then create a list (rows).
        root = tree.getroot()
        record_list = [x.attrib for x in root.iter('Record')]
        # create DataFrame from a list (rows) of dictionaries (columns)
        data = pd.DataFrame(record_list)
        # proper type to dates
        for col in ['creationDate', 'startDate', 'endDate']:
            data[col] = pd.to_datetime(data[col])
        # value is numeric, NaN if fails
        data['value'] = pd.to_numeric(data['value'], errors='coerce')
        # some records do not measure anything, just count occurences
        # filling with 1.0 (= one time) makes it easier to aggregate
        data['value'] = data['value'].fillna(1.0)
        #data['value']=data['value'].replace(0, np.nan)
        # shorter observation names: use vectorized replace function
        data['type'] = data['type'].str.replace('HKQuantityTypeIdentifier', '')
        data['type'] = data['type'].str.replace('HKCategoryTypeIdentifier', '')
        # pivot and resample
        agg_dict = {
            #'Height':np.mean,
            #'BodyMass':np.mean,
            'HeartRate':np.mean,
            'StepCount':np.sum,
            'DistanceWalkingRunning':np.sum,
            'BasalEnergyBurned':np.sum,
            'ActiveEnergyBurned':np.sum,
            'FlightsClimbed':np.sum,
            'AppleExerciseTime':np.sum,
            'RestingHeartRate':np.mean,
            'VO2Max':np.mean,
            'WalkingHeartRateAverage':np.mean,
            #'HeadphoneAudioExposure':np.sum,
            #'SixMinuteWalkTestDistance':np.mean,
            'AppleStandTime':np.sum,
            #'HKDataTypeSleepDurationGoal':np.mean,
            #'SleepAnalysis':np.sum,
            'AppleStandHour':np.sum,
            #'MindfulSession',
            'HeartRateVariabilitySDNN':np.mean
        }
        pivot_df = data.pivot_table(index='endDate', columns=['type'], values='value')
        df = pivot_df.resample('D').agg(agg_dict)
        df=df.replace(0, np.nan)
        return df

    try:
        #df = get_data()
        with st.spinner('HealthKit export.xml file processing could take a time... Wait for it...'):
            df = get_data()
        st.success('Done!')
        records = st.multiselect(
            "Choose records", list(df), ['BasalEnergyBurned','ActiveEnergyBurned','AppleExerciseTime','AppleStandHour', 'HeartRateVariabilitySDNN']
        )
        #st.write(records)
        if not records:
            st.error("Please select at least one record.")
        else:
            data = df[records]
            #data /= 1000000.0
            st.write("## Data", data)
            st.write("## Data description", data.describe())
            st.line_chart(data)
            
            with st.spinner('Plotting pairplot... Wait for it...'):
                #fig = plt.figure(figsize=(8,6)) 
                fig = sns.pairplot(data) 
                # add observation dots
                #g.map_offdiag(sns.scatterplot, marker='.', color='black')
                st.pyplot(fig)
            st.success('Done!')
                            
            with st.spinner('Plotting heatmap... Wait for it...'):
                # correlation matrix
                cm = data.corr()
                # heatmap
                fig = plt.figure(figsize=(8,6)) 
                sns.heatmap(cm, annot=True, fmt=".2f", vmin=-1.0, vmax=+1.0, cmap='Spectral')
                st.pyplot(fig)
            st.success('Done!')

    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**
            Connection error: %s
        """
            % e.reason
        )


st.set_page_config(page_title="HealthKit Demo", page_icon="📊")
st.markdown("# HealthKit Demo")
st.sidebar.header("HealthKit Demo")
st.write(
    """This demo shows how to analyze HealthKit File.
(--------)"""
)

health_kit_demo()

show_code(health_kit_demo)
