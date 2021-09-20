# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import streamlit as st
from numpy import column_stack


def print_hi(name):
	# Use a breakpoint in the code line below to debug your script.
	print(f'Hi, {name}')  # Press F9 to toggle the breakpoint.

@st.cache
def populate_nans(dataframe):
	"""
	за всички NaN полета задаваме стойност 0
	"""
	dataframe = dataframe.fillna(0)
	return dataframe

@st.cache(suppress_st_warning=True)
def equalize_events(dataframe):
	events_count = dataframe.EventId.value_counts()  # или dataframe['EventId'].value_counts(),
													# (почти никога) няма значение

	# type(events_count) резултата е : <class 'pandas.core.series.Series'>
	st.write(f"\nТрябва ни размер на матрицата не по-малък, и не по-голям от {events_count.min()} реда.")
	st.write(events_count)

	count_class_33024, count_class_33025, count_class_33026 = dataframe.EventId.value_counts()

	# Divide by class
	df_class_33024 = dataframe[dataframe['EventId'] == 33024]
	df_class_33025 = dataframe[dataframe['EventId'] == 33025]
	df_class_33026 = dataframe[dataframe['EventId'] == 33026]
	# print(df_class_33024.shape,df_class_33025.shape,df_class_33026.shape,sep="\t")
	# dataframe.loc[dataframe['EventId'] == 33025]
	# st.write(dataframe.iloc[435:445]) # Ако погледнем към колоната с ивентите, виждаме къде точно започва следващия
	# st.write(df_class_33024)

	df_class_33024_undersample = df_class_33024.sample(count_class_33025)
	df_alex_under = pd.concat([df_class_33024_undersample, df_class_33025,df_class_33026], axis=0) # това е тест сет

	# st.write('Random under-sampling:')
	# st.write(df_alex_under.EventId.value_counts())

	return df_alex_under

@st.cache
def preprocess():
	"""load, populate, equalize"""
	df = load_df(
		'data/4_GD_14ch_LR_24_01_2020_VStim_20Repeat_3sec_epoch_3_off_0.25_chunk_0.025_4_with_shunk_and_notnull.csv',
		["AF3", "F7", "F3", "F5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4", "EventId"]
	)
	if 'is_df_loaded' not in st.session_state:
		st.session_state['is_df_loaded'] = True

	df = populate_nans(df)
	# Initialization
	if 'populated_nans_' not in st.session_state:
		st.session_state['populated_nans_'] = True

	df = equalize_events(df)
	# Initialization
	if 'equalized_events_' not in st.session_state:
		st.session_state['equalized_events_'] = True

	return df


@st.cache
def process(dataframe):
	# dataframe.to_csv('processed.csv', encoding='utf8')
	processed_df = pd.read_csv('processed.csv')
	processed_sorted = processed_df.sort_values(by=['Unnamed: 0'])
	preprocessed_sorted = processed_sorted.reset_index()
	processed = preprocessed_sorted.drop(columns=['Unnamed: 0', 'index'])
	# st.dataframe(processed)
	# Initialization
	if 'processed_' not in st.session_state:
		st.session_state['processed_'] = True
	return processed

@st.cache
def load_in_cache(df):
	return df

@st.cache
def convert_df(df):
	return df.to_csv(index=False,encoding='utf-8')  # .encode('utf-8')


def run_app():
	# df = pd.read_csv()
	col1,col2 = st.columns([1,1])
	with col1:
		st.write([k for k in st.session_state.keys()])
	with col2:
		st.write([v for v in st.session_state.values()])

	df = preprocess()
	st.info('`preprocess()`')

	df = process(df)
	st.info('`process(df)`')

	df = load_in_cache(df)
	st.info('`load_in_cache(df`')
	st.write(df.head(10))
	# s = df.shape
	# print((s[0]-1) / 20)
	# st.write(df.iloc[:20])
	step_size = 20
	iters = int(df.shape[0]/step_size)
	features = df.drop(['EventId'],axis=1)
	# print(type(features))
	# st.code(features)
	seq = []
	seq2 = []
	x2 = int()
	for x in range(0,iters,step_size):
		seq.append(x)
		x2 = x +20
		seq2.append(x2)
		# print(features.iloc[:x])
		# print(f'las hiks: {x2}\nhiks: {x}')
	# print(seq)
	# print(seq2)
	y = tuple(zip(seq,seq2))
	# print(y)
	final_df = pd.DataFrame(columns=["AF3", "F7", "F3", "F5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4", "EventId"])
	for me in enumerate(y):
		otdo = me[1]
		# print(f"\t{me[0]}\n{df.iloc[otdo[0]:otdo[1]].mean()}")
		epoch20 = df.iloc[otdo[0]:otdo[1]]
		avg_epoch20 = epoch20.mean()
		# print(type(avg_epoch20))
		final_df = final_df.append(avg_epoch20, ignore_index=True)
		# print(avg_rows)
	st.header("final df")
	st.dataframe(final_df)
	col1_1,col2_1 = st.columns([1,1])
	with col1_1:
		st.download_button(
			"Press to Download",
			convert_df(final_df),
			"browser_visits.csv",
			"text/csv",
			key='browser-data'
		)

	with col2_1:
		st.button('cls')

	# Initialization
	if 'df_name' not in st.session_state:
		st.session_state['df_name'] = 'epochs'

	# final_df.name = "epochs"
	# final_df.to_csv('final_dataset.csv',index=False,encoding='utf8')
	# convert_df(final_df)
	# Initialization
	if 'df_name' not in st.session_state:
		st.session_state['df_name'] = 'epochs'

	st.write(final_df.EventId.astype(int).unique())
	st.write(final_df.EventId.value_counts())

@st.cache
def load_df(csv_file,default_columns):
	df = pd.read_csv(csv_file)
	df = df[default_columns]
	return  df # pd.read_csv(csv_file)


def save_to_pickle(df):
	df.to_pickle('step1.pkl')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	# Session State also supports attribute based syntax
	st.code('start here')
	x = tuple(zip(st.session_state.keys(),st.session_state.values()))
	print(x)
	if ('is_df_loaded' in st.session_state) & ('populated_nans_' in st.session_state) & ('equalized_events_' in st.session_state):
		print('all 1')
	if ('is_df_loaded' in st.session_state) | ('populated_nans_' in st.session_state) | ('equalized_events_' in st.session_state):
		print('atleast 1')

	# 	st.session_state.df_name = 'epochs'
	# else:
	# 	st.code('here')


	# # Page expands to full width to support big screens
	# st.set_page_config(page_title='The Machine Learning App', layout='wide')
	run_app()
	if 'df_name' in st.session_state:
		st.code(f'after here {st.session_state.df_name}')
		# st.session_state.df_name = 'epochs'
	if 'is_df_loaded' in st.session_state:
		st.code(f'after here {st.session_state.is_df_loaded}')
		# st.session_state.df_name = 'epochs'

	print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
