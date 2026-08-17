#%%
import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)
import preprocessing, models


################################################################################

target_feature = 'LF98TRUU_Index_OAS'
momentum_list = ['LF98TRUU_Index_OAS', 'LUACTRUU_Index_OAS']
file_buffer =  './data/Economic_Data_2020_08_01.xlsx'

pipeline = preprocessing.BloombergPreprocessor(file_buffer,
                                           target_feature,
                                           momentum_list = momentum_list
                                           )

new_model = models.MomentumModel(pipeline, model_name='XGBoost')

#works!
new_model.predictive_power()
new_model.feature_importance()
new_model.feature_importance_over_time(forecast_range=30)
new_model.get_mean_error_metrics()

# Needs a classifier not binary model
#new_model.get_roc_and_precision_recall_curves()

# %%
