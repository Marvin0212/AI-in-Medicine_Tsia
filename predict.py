# -*- coding: utf-8 -*-
"""predict_own_stack_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o5MdMAEy4KX56ChhJIJ0XPY7BwGovblu
"""

# -*- coding: utf-8 -*-
"""predictlabels_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17LGCa14OwtKXs5uq5VgPx2v121coWIHa
"""

# -*- coding: utf-8 -*-
"""Explore_Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18G7sZ51hHhf7tbhx_jmexwXFlgJiK8mV
"""

# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle

def f1_score(y_true, y_pred):
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

##preprocessing of data##########
def data_preprocessing_short(ecg_lead):
        desired_length=3000
        amount_data, remainder=divmod(ecg_lead.shape[0],desired_length)
        if remainder!=0:
          # Loop the remainder part until it has the desired length
          looped_remainder = np.tile(ecg_lead[desired_length*amount_data:], (desired_length // remainder + 1))[:desired_length]
          # Attach the looped remainder back to the ecg_lead
          ecg_array_processed = np.concatenate((ecg_lead[0:desired_length*amount_data], looped_remainder))
          amount_data+=1
          return ecg_array_processed,amount_data     
        else:
          return ecg_lead,amount_data
###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

#------------------------------------------------------------------------------
    predictions = list()   
    #load pretrained scaler
    #std_slc = pickle.load(open('/content/drive/My Drive/scaler.pkl','rb'))
    std_slc = pickle.load(open('minmax_scaler_time.pkl','rb'))
    std_slc_fft = pickle.load(open('std_scaler_fft.pkl','rb'))
    list_len = np.array(ecg_leads)
    data_lengths = np.empty(len(list_len),int)
    ecg_data = np.empty((1,3000))
    ecg_data_fft = np.empty((1,1500))
    for idx,ecg_lead in enumerate(ecg_leads):
########preprocessing
        #loop array
        line_count = 0
        desired_length=3000
        ecg_lead = ecg_lead.astype('float32')
        ecg_array_processed, amount_data= data_preprocessing_short(ecg_lead) #data preprocessing 
        data_lengths[idx]=int(amount_data)
        amount_data=np.arange(0,amount_data)  
        for i in amount_data: 
          if line_count != 0:                                        #check if its the first time so that vstack can work properly
            ecg_arrays = np.vstack((ecg_arrays,ecg_array_processed[i*desired_length:(i+1)*desired_length]))   
          else:
            ecg_arrays=ecg_array_processed[0:desired_length]
          line_count = line_count + 1 
        ecg_arrays_processed = ecg_arrays
        if(len(ecg_arrays_processed)!=3000):
          for i, series in enumerate(ecg_arrays):
            #clip data
            data=ecg_arrays_processed[i,:]
            ecg_arrays_processed[i,:] = np.clip(data, -500, 500)
            data= data.reshape(1, -1)
            ecg_arrays_processed[i,:]=std_slc.transform(data)
            ecg_data = np.vstack((ecg_data,ecg_arrays_processed[i,:]))
        else:
          print(ecg_arrays_processed.shape,'ecg_arrays_processed.shape data')
          ecg_arrays_processed = np.clip(ecg_arrays_processed, -500, 500)
          ecg_arrays_processed= ecg_arrays_processed.reshape(1, -1)
          ecg_arrays_processed=std_slc.transform(ecg_arrays_processed)
          ecg_data=np.vstack((ecg_data,ecg_arrays_processed))
        
########FFT
        if(len(ecg_arrays)!=3000):
          ecg_array_fft = np.empty((ecg_arrays.shape[0], 1500))
          for i, series in enumerate(ecg_arrays):
            fft_results = np.fft.fft(ecg_arrays[i,:], 3000)
            # Only save the positive half of the FFT results
            ecg_array_fft[i,:] = np.abs(fft_results[:1500])
            data=ecg_array_fft[i,:]
            data= data.reshape(1, -1)
            ecg_array_fft[i,:]=std_slc_fft.transform(data)
            ecg_data_fft = np.vstack((ecg_data_fft,ecg_array_fft[i,:]))   
        else:
          print(ecg_arrays.shape,'ecg_arrays_processed.shape fft')
          ecg_array_fft = np.empty((0, 1500))
          fft_results = np.fft.fft(ecg_arrays, 3000)
          # Only save the positive half of the FFT results
          ecg_array_fft = np.abs(fft_results[:1500])
          ecg_array_fft= ecg_array_fft.reshape(1, -1)
          ecg_array_fft=std_slc_fft.transform(ecg_array_fft)
          ecg_data_fft = np.vstack((ecg_data_fft,ecg_array_fft))
        
########make predictions
    print(ecg_data.shape[0],'data length')
    print(ecg_data_fft.shape[0],'fft data length')
    def predict_time(model_names):
        data_length=ecg_data.shape[0]
        # list to store the predictions of the base models
        pred_Time = np.empty((data_length, 2*16 ))
        for i, model_name in enumerate(model_names):
            # load the model
            model = load_model('Models/'+model_name,custom_objects={'f1_score': f1_score})
            # get the predictions for the train set
            pred_Time[:,i*2:(i*2)+2]=model.predict(ecg_data,verbose=0)
        return pred_Time
    time_models = ['best_model_time0.h5', 'best_model_time1.h5', 'best_model_time2.h5', 'best_model_time3.h5',
                  'best_model_time4.h5', 'best_model_time5.h5', 'best_model_time6.h5', 'best_model_time7.h5',
                  'best_model_time8.h5', 'best_model_time9.h5', 'best_model_time10.h5', 'best_model_time11.h5',
                  'best_model_time12.h5','best_model_time13.h5','best_model_time14.h5', 'best_model_time15.h5']
    pred_Time = predict_time(time_models)  
    def predict_fft(model_names):
        data_length=ecg_data_fft.shape[0]
        pred_fft = np.empty((data_length, 2*16 ))
        for i, model_name in enumerate(model_names):
            # load the model
            model = load_model('Models/'+model_name,custom_objects={'f1_score': f1_score})#'Models/'+
            # get the predictions for the train set
            pred_fft[:,i*2:(i*2)+2]=model.predict(ecg_data_fft,verbose=0)
        return pred_fft
    fft_models = ['best_model_fft0.h5', 'best_model_fft1.h5', 'best_model_fft2.h5', 'best_model_fft3.h5',
                  'best_model_fft4.h5', 'best_model_fft5.h5', 'best_model_fft6.h5', 'best_model_fft7.h5',
                  'best_model_fft8.h5', 'best_model_fft9.h5', 'best_model_fft10.h5', 'best_model_fft11.h5',
                  'best_model_fft12.h5','best_model_fft13.h5','best_model_fft14.h5', 'best_model_fft15.h5']
    pred_fft= predict_fft(fft_models)
    pred_fft = pred_fft.reshape(pred_fft.shape[0], pred_fft.shape[1], 1)
    pred_Time = pred_Time.reshape(pred_Time.shape[0], pred_Time.shape[1], 1)

    model = load_model('Models/best_stack_model0.87192.h5',custom_objects={'f1_score': f1_score})
    stack_predictions=model.predict([pred_Time,pred_fft],verbose=0) 
    print(stack_predictions.shape)
    j=1
    for i in range(len(list_len)):
      prediction = np.sum(stack_predictions[j:data_lengths[i]+j+1,:], axis=0) / data_lengths[i] 
      j+=data_lengths[i]
      if prediction[0] > 0.5:
          predictions.append((ecg_names[i], 'N'))
      else:
          predictions.append((ecg_names[i], 'A'))
    # if ((idx+1) % 100)==0:
    print(str(idx+1) + "\t Dateien wurden verarbeitet.")          
#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!