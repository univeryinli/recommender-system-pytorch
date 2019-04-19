# -*- coding:utf-8 -*-

from utils import data_preprocess
from model import DeepFM
import torch
import pickle

train_dict,test_dict = data_preprocess.read_csv_dataset('./data/final_track2_train_new.csv',task='finish')
#pred_dict=data_preprocess.read_csv_dataset_pred('./data/small_test.csv',task='like')
#train_dict = data_preprocess.read_criteo_data('./data/tiny_train_input.csv', './data/category_emb.csv')
#test_dict = data_preprocess.read_criteo_data('./data/tiny_test_input.csv', './data/category_emb.csv')

deepfm = DeepFM.DeepFM(8,train_dict['feature_sizes'],verbose=True,use_cuda=True, weight_decay=0.0001,use_fm=True,use_ffm=False,use_deep=False)
#pred=deepfm.predict_from_model_file(pred_dict['index'], pred_dict['value'],deepfm,'./saved_model')
#pred=deepfm.predict(pred_dict['index'], pred_dict['value'])
#pickle.dump(pred,open('like_pre','wb'))
deepfm.fit(train_dict['index'], train_dict['value'], train_dict['lable'],test_dict['index'], test_dict['value'], test_dict['lable'],ealry_stopping=True,refit=False,save_path='./saved_model')
