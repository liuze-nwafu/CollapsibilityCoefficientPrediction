#!/usr/bin/env python
# -*- coding: utf-8 -*-

import wx
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import math
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
#from xgboost import XGBRegressor
#from deepforest import main
#from catboost import CatBoostRegressor
import pandas as pd
import math
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os,sys
feature_selection_function=0
classifier_selection_function=0
feature_validation_function=1

path_2='No test data specified'
class pssm_panel(wx.Panel):
    def __init__(self, parent, id):
        wx.Panel.__init__(self, parent, id)

        #特征选择
        listSelection = [u"Use all features(&U)", u"Use some important features(&D)"]
        self.radio_FeatureSelection = wx.RadioBox(self, -1, u"Feature selection", (50, 30), (680, 150), listSelection, 2, wx.RA_SPECIFY_ROWS)
        self.radio_FeatureSelection.Bind(wx.EVT_RADIOBOX, self.feature_selection)
        wx.StaticText(self, label="Number of input features：", pos = (80, 130))
        self._FeatureNum = wx.TextCtrl(self, pos = (180, 130))
        self._FeatureNum.Disable()
        wx.StaticText(self, label="Explanation: Please choose between 1 and 3", pos = (330, 135))
        
        self.Bind(wx.EVT_PAINT, self.OnPaint)

        #分类器选择
        #listSelection_classifier = [u"SVM(&S)", u"Randomforest(&F)", u"Adaboost(&A)", u"CatBoost(&X)", u"Deepforest(&C)"]
        listSelection_classifier = [u"DecisionTree(&S)", u"RandomForest(&F)", u"SVR(&A)", u"LGBM(&X)",u"xGBoost(&C)"]
        self.radio_FeatureSelection_classifier = wx.RadioBox(self, -1, u"Regressor selection ", (50, 230), (680, 150), listSelection_classifier, 5, wx.RA_SPECIFY_ROWS)
        self.radio_FeatureSelection_classifier.Bind(wx.EVT_RADIOBOX, self.classifier_selection)
        #self.C=wx.TextCtrl(self,-1,'0',pos=(320,250))
        # wx.StaticText(self, label="Please input C:", pos = (250, 252))
        # self.gamma=wx.TextCtrl(self,-1,'0',pos=(570,250))
        # wx.StaticText(self, label="Please input γ:", pos = (500, 252))
        # self.random_state=wx.TextCtrl(self,-1,'0',pos=(320,280))
        # wx.StaticText(self, label="random_state:", pos = (195, 282))
        # self.n_estimators=wx.TextCtrl(self,-1,'100',pos=(570,280))

        self.max_depth = wx.TextCtrl(self, -1, '0', pos=(320, 250))
        wx.StaticText(self, label="max_depth:", pos = (195, 252))
        self.min_samples_leaf=wx.TextCtrl(self,-1,'0',pos=(570,250))
        wx.StaticText(self, label="min_samples_leaf:", pos = (458, 252))
        self.min_samples_split=wx.TextCtrl(self,-1,'0',pos=(320,280))
        wx.StaticText(self, label="min_samples_split:", pos = (195, 282))
        self.n_estimators=wx.TextCtrl(self,-1,'100',pos=(570,280))
        wx.StaticText(self, label="n_estimators:", pos = (460, 282))
        for eachText in [self.max_depth,self.min_samples_leaf,self.min_samples_split,self.n_estimators]:
            eachText.Enable(False)

        #按钮构建模型
        note=' Explanation: To ensure prediction accuracy, if there are no special requirements\n\t\tit is recommended to use all features and construct the model \n\t\tusing the default parameters of the regression model.'
        wx.StaticText(self, label=note, pos = (70, 480))
        wx.StaticText(self, label="Model construction", pos = (60, 420))
        self.button_trainmodel1 = wx.Button(self, -1, u"Model training", (120, 580), (100, 40))
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.button_trainmodel1)
        #模型构建进度
        wx.StaticText(self, label="progress:", pos = (410,590))
        self.gauge = wx.Gauge(self, range=100, size=(190, -1), pos=(470,590))
        self.ModelState=wx.StaticText(self, label='0%', pos = (670, 590), size=(40,20))

        #按钮重置模型
        self.resetmodel1 = wx.Button(self, -1, u"Reset", (250, 580), (100, 40))
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.resetmodel1)
        self.resetmodel1.Disable()
        
        ##################################################################第二页
        
        #导入PSSM路径
        wx.StaticText(self, label="Test sample :", pos = (770, 50))
        self.buttonimport1 = wx.Button(self, -1, u"browse", (1350, 40), (100, 40))
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.buttonimport1)
        self.buttonimport1.Disable()
        self.FileState=wx.TextCtrl(self,-1,path_2,pos=(920,45),size=(400,35))
        self.FileState.Disable()
        #提交和清除预测
        self.buttonSubmit1 = wx.Button(self, -1, u"Submit", (1300, 160), (100, 40))
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.buttonSubmit1)
        self.buttonSubmit1.Disable()
        #预测结果
        wx.StaticText(self, label="Results", pos = (810, 120))
        wx.StaticText(self, label="Predicted Value:", pos = (850, 195))
        self.prediction_pressure=wx.StaticText(self, label='', pos = (1020, 195))
        self.prediction_reset = wx.Button(self, -1, u"Clear data", (1300, 210), (100, 40))
        self.Bind(wx.EVT_BUTTON, self.OnClick_prediction, self.prediction_reset)
        self.prediction_reset.Disable()
        
        #模型检验
        wx.StaticText(self, label="model validation", pos = (810, 320))
        self.validation1 = wx.RadioButton(self, label=u"Independent test(&I)", pos=(1100, 360), style=wx.RB_GROUP)

        self.validation1.Bind(wx.EVT_RADIOBUTTON, self.Set_validation_1)

        #模型检验按键
        self.validation = wx.Button(self, -1, u"Start validation", (1000, 400), (120, 60))
        self.Bind(wx.EVT_BUTTON, self.OnClick_validation, self.validation)
        self.validation.Disable()
        self.validation_reset = wx.Button(self, -1, u"Clear data", (1200, 400), (120, 60))
        self.Bind(wx.EVT_BUTTON, self.OnClick_validation, self.validation_reset)
        self.validation_reset.Disable()

        #模型检验结果
        wx.StaticText(self, label="MAE:", pos = (900, 505))
        wx.StaticText(self, label="MSE:", pos = (1200, 505))
        wx.StaticText(self, label="Median:", pos = (900, 555))
        wx.StaticText(self, label="R2:", pos = (1200, 555))

        self.MAE=wx.StaticText(self, label='', pos = (1000, 505))
        self.MSE=wx.StaticText(self, label='', pos = (1300, 505))
        self.Median=wx.StaticText(self, label='', pos = (1000, 555))
        self.R2=wx.StaticText(self, label='', pos = (1300, 555))

    def Set_validation_1(self, e):
        global feature_validation_function
        feature_validation_function=1
    def Set_validation_2(self, e):
        global feature_validation_function
        feature_validation_function=2
        
    def absolute_path(self,relative):
        if hasattr(sys,"_MEIPASS"):
            absolute_path1 = os.path.join(sys._MEIPASS,relative)
        else:
            absolute_path1 = os.path.join(relative)
        return absolute_path1

    def OnPaint(self, e):
        dc = wx.PaintDC(self)
        dc.SetPen(wx.Pen('#C7C3C3'))
        brush1 = wx.Brush(wx.Bitmap(self.absolute_path("image\\界面1.jpg")))
        #brush1 = wx.Brush(wx.Bitmap('E:\\tumu\image\界面1.jpg'))
        dc.SetBrush(brush1)
        dc.DrawRectangle(0,0,1920,1080)
        
        dc.SetBrush(wx.Brush(wx.Colour(255,255,255)))
        dc.DrawLine(750,0,750,700)
        #路径输入框
        dc.DrawRectangle(1020, 45, 300, 35)
        #右侧最大框
        dc.DrawRectangle(800, 130, 700, 150)
        dc.DrawRectangle(800, 330, 700, 320)
        #左下框
        dc.DrawRectangle(50, 430, 680, 220)
        #模型检验文本边框
        dc.DrawRectangle(980, 500, 200, 33)
        dc.DrawRectangle(980, 550, 200, 33)
        dc.DrawRectangle(1280, 500, 200, 33)
        dc.DrawRectangle(1280, 550, 200, 33)
        #预测部分文本边框
        dc.DrawRectangle(1000, 185, 200, 33)

        
    def feature_selection(self,event):
        global feature_selection_function
        if self.radio_FeatureSelection.GetSelection()==0 :
            feature_selection_function=0
            self._FeatureNum.Disable()
        if self.radio_FeatureSelection.GetSelection()==1 :
            feature_selection_function=1
            self._FeatureNum.Enable()
    
    def classifier_selection(self,event):#事件处理器
        global classifier_selection_function
        if self.radio_FeatureSelection_classifier.GetSelection()==0 :
            classifier_selection_function=0
            # for eachText in [self.C,self.gamma]:
            #     eachText.Enable(True)

            for eachText in [self.max_depth, self.min_samples_leaf, self.min_samples_split, self.n_estimators]:
                eachText.Enable(False)
        elif self.radio_FeatureSelection_classifier.GetSelection()==1 :
            classifier_selection_function=1
            # for eachText in [self.C,self.gamma]:
            #     eachText.Enable(False)
            for eachText in [self.max_depth, self.min_samples_leaf, self.min_samples_split, self.n_estimators]:
                eachText.Enable(True)
        elif self.radio_FeatureSelection_classifier.GetSelection()==2 :
            classifier_selection_function=2
            # for eachText in [self.C,self.gamma]:
            #     eachText.Enable(False)
            for eachText in [self.max_depth, self.min_samples_leaf, self.min_samples_split, self.n_estimators]:
                eachText.Enable(False)
        elif self.radio_FeatureSelection_classifier.GetSelection()==3 :
            classifier_selection_function=3
            # for eachText in [self.C,self.gamma]:
            #     eachText.Enable(False)
            for eachText in [self.max_depth, self.min_samples_leaf, self.min_samples_split, self.n_estimators]:
                eachText.Enable(False)
        else:
            classifier_selection_function=4
            # for eachText in [self.C,self.gamma]:
            #     eachText.Enable(False)
            for eachText in [self.max_depth, self.min_samples_leaf, self.min_samples_split, self.n_estimatorss]:
                eachText.Enable(False)

    def trainmodel(self):
        global clf, feature, pressure

        #feature = pd.read_csv('dataset.csv',encoding='gbk')
        feature=pd.read_excel('dataCC.xlsx')

        pressure = feature["湿陷系数"]

        pressure_array = pressure

        feature.drop("湿陷系数", axis=1, inplace=True)

        min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-1, 1))
        feature = min_max_scaler.fit_transform(feature)

        feature_array = feature

        if feature_selection_function==1:
            global num1_1
            num1_1=int(self._FeatureNum.GetValue())
            feature_array=feature[...,0:int(num1_1)]

        X, X_validation, y, y_validation = train_test_split(feature_array, pressure_array,
                                                            test_size=0.25,random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.25, random_state=42)

        if classifier_selection_function==0:
            #clf=svm.SVR(kernel='rbf',C=float(self.C.GetValue()),gamma=float(self.gamma.GetValue()))
            clf = DecisionTreeRegressor(max_depth=8,min_samples_split=2,min_samples_leaf=1)
        elif classifier_selection_function==1:
            clf=RandomForestRegressor(n_estimators=int(self.n_estimators.GetValue()),max_depth=int(self.max_depth.GetValue()),
                                      min_samples_leaf=int(self.min_samples_leaf.GetValue()),min_samples_split=int(self.min_samples_split.GetValue()))

        elif classifier_selection_function==2:
            clf = svm.SVR(C=100,epsilon=0.026,gamma=0.503)
        elif classifier_selection_function==3:
            clf =LGBMRegressor(num_leaves=20,
    learning_rate = 0.03,
    n_estimators = 629,
    min_data_in_leaf = 10,
    max_depth = 7)
        else:
            clf =XGBRegressor(
        learning_rate=0.1,
        n_estimators=470,
        max_depth = 10,
        min_child_weight = 9
    )

        clf=clf.fit(X_train,y_train)
        
    def OnClick_validation(self, event):
        if event.GetEventObject() == self.validation :
            if feature_validation_function==1:
                pssm_panel.independent_test(self)
            self.validation_reset.Enable()
        else:
            self.validation_reset.Disable()
            self.MSE.SetLabel('')
            self.MAE.SetLabel('')
            self.Median.SetLabel('')
            self.R2.SetLabel('')
    def OnClick_prediction(self, event):
        self.prediction_reset.Disable()
        self.prediction_pressure.SetLabel('')

    def independent_test(self):

        if feature_selection_function==0:
            global num1_1
            feature_all=feature


        if feature_selection_function==1:
            global num1_1
            num1_1=int(self._FeatureNum.GetValue())
            feature_all=feature[...,0:int(num1_1)]

        X, X_validation, y, y_validation = train_test_split(feature_all, pressure,test_size=0.25, random_state=0)

        y_train = y

        X_train = X

        X_test = np.array(X_validation, dtype=np.float32)
        y_test = y_validation

        score_r=clf.score(X_train,y_train)
        y_pred = clf.predict(X_test)

        MAE = metrics.mean_absolute_error(y_test, y_pred)
        MSE = metrics.mean_squared_error(y_test, y_pred)
        Median = metrics.median_absolute_error(y_test, y_pred)
        R2 = metrics.r2_score(y_test, y_pred)

        self.MAE.SetLabel(str(MAE))
        self.MSE.SetLabel(str(MSE))
        self.Median.SetLabel(str(Median))
        self.R2.SetLabel(str(R2))

    ## 每个按钮的具体行为
    def OnClick(self, event):
        if event.GetEventObject() == self.buttonimport1 :
            dlg = wx.FileDialog(self,u"Input file",style=wx.DD_DEFAULT_STYLE)
            if dlg.ShowModal() == wx.ID_OK:
                global path_2
                path_2=dlg.GetPath()
                dlg.Destroy()
            self.FileState.SetLabel(path_2)
            self.buttonSubmit1.Enable()
        elif event.GetEventObject() == self.buttonSubmit1 :
            pssm_open=0
            if path_2.endswith(".csv"):

                    pssm_panel.predict_pssm(self)
                    pssm_open=1
            if pssm_open==0:
                dlg_7 = LoginDialog_7(None,-1)
                dlg_7.ShowModal()
                dlg_7.Destroy()

        elif event.GetEventObject() == self.button_trainmodel1 :
            m=0
            if feature_selection_function==1 :
                if self._FeatureNum.GetValue()=='':
                    dlg = LoginDialog(None,-1)
                    dlg.ShowModal()
                    dlg.Destroy()
                    m=1
                elif self._FeatureNum.GetValue().isdigit()==0:
                    dlg_2 = LoginDialog_2(None,-1)
                    dlg_2.ShowModal()
                    dlg_2.Destroy()
                    m=1
                elif int(self._FeatureNum.GetValue())<=0 or int(self._FeatureNum.GetValue())>=9:
                    dlg_3 = LoginDialog_3(None,-1)
                    dlg_3.ShowModal()
                    dlg_3.Destroy()
                    m=1
            if classifier_selection_function==1:
                if self.n_estimators.GetValue()=='' or self.max_depth.GetValue()==''or self.min_samples_leaf.GetValue()==''or self.min_samples_split.GetValue()=='':
                    dlg_4 = LoginDialog_4(None,-1)
                    dlg_4.ShowModal()
                    dlg_4.Destroy()
                    m=1
                elif self.n_estimators.GetValue().isdigit()==0 or self.max_depth.GetValue().isdigit()==0:
                    dlg_5 = LoginDialog_5(None,-1)
                    dlg_5.ShowModal()
                    dlg_5.Destroy()
                    m=1
            # if classifier_selection_function==0:
            #     a=self.C.GetValue().replace('.','',1)
            #     b=self.gamma.GetValue().replace('.','',1)
            #     if a.isdigit()==0 or b.isdigit()==0:
            #         dlg_6 = LoginDialog_6(None,-1)
            #         dlg_6.ShowModal()
            #         dlg_6.Destroy()
            #         m=1
            if m !=1:
                self.button_trainmodel1.Disable()
                pssm_panel.trainmodel(self)
                self.gauge.SetValue(100)
                self.ModelState.SetLabel('100%')
                self.buttonimport1.Enable()
                self.validation.Enable()
                self.resetmodel1.Enable()
                self.FileState.Enable()
                
        elif event.GetEventObject() == self.resetmodel1 :
            self.resetmodel1.Disable()
            self.gauge.SetValue(0)
            self.ModelState.SetLabel('0%')
            self.button_trainmodel1.Enable()
            self.MAE.SetLabel('')
            self.MSE.SetLabel('')
            self.Median.SetLabel('')
            self.R2.SetLabel('')
            self.buttonimport1.Disable()
            self.buttonSubmit1.Disable()
            self.validation_reset.Disable()
            self.prediction_reset.Disable()
            self.validation.Disable()
            self.FileState.Disable()
            
        else:
            print("No button is clicked.")
            
    def predict_pssm(self):

        validation = pd.read_csv(path_2)

        validation = np.array(validation)

        if feature_selection_function==1:
            global num1_1
            validation=validation[...,0:int(num1_1)]

        predict_y_test = clf.predict(validation)

        self.prediction_pressure.SetLabel(str(predict_y_test[0]))
            #wx.StaticText(self, label=str(prob_predict_y_test[0][1]), pos = (1220, 390))
        self.prediction_reset.Enable()


class LoginDialog(wx.Dialog):
    def __init__(self, parent, id):
        super(LoginDialog, self).__init__(parent, id, u'Missing information',size=(200,200))
        self.sizer1 = wx.BoxSizer(wx.VERTICAL)
        self.sizer1.Add(wx.StaticText(self, -1, u"Number of features not entered"),
                         0, wx.ALIGN_CENTER|wx.TOP, border=50)
        self.sizer1.Add(wx.Button(self, wx.ID_OK), 0, wx.ALIGN_CENTER_HORIZONTAL|wx.TOP|wx.BOTTOM, border=10 )
        self.SetSizer(self.sizer1)
        self.Centre()
        
class LoginDialog_2(wx.Dialog):
    def __init__(self, parent, id):
        super(LoginDialog_2, self).__init__(parent, id, u'Information error',size=(200,200))
        self.sizer1 = wx.BoxSizer(wx.VERTICAL)
        self.sizer1.Add(wx.StaticText(self, -1, u"The number of features is not an integer"),
                         0, wx.ALIGN_CENTER|wx.TOP, border=50)
        self.sizer1.Add(wx.Button(self, wx.ID_OK), 0, wx.ALIGN_CENTER_HORIZONTAL|wx.TOP|wx.BOTTOM, border=10 )
        self.SetSizer(self.sizer1)
        self.Centre()

        
class LoginDialog_3(wx.Dialog):
    def __init__(self, parent, id):
        super(LoginDialog_3, self).__init__(parent, id, u'Information error',size=(300,200))
        self.sizer1 = wx.BoxSizer(wx.VERTICAL)
        self.sizer1.Add(wx.StaticText(self, -1, u"The number of features must be between 1 and 3"),
                         0, wx.ALIGN_CENTER|wx.TOP, border=50)
        self.sizer1.Add(wx.Button(self, wx.ID_OK), 0, wx.ALIGN_CENTER_HORIZONTAL|wx.TOP|wx.BOTTOM, border=10 )
        self.SetSizer(self.sizer1)
        self.Centre()

        
class LoginDialog_4(wx.Dialog):
    def __init__(self, parent, id):
        super(LoginDialog_4, self).__init__(parent, id, u'Information missing',size=(200,200))
        self.sizer1 = wx.BoxSizer(wx.VERTICAL)
        self.sizer1.Add(wx.StaticText(self, -1, u"Missing random forest parameters"),
                         0, wx.ALIGN_CENTER|wx.TOP, border=50)
        self.sizer1.Add(wx.Button(self, wx.ID_OK), 0, wx.ALIGN_CENTER_HORIZONTAL|wx.TOP|wx.BOTTOM, border=10 )
        self.SetSizer(self.sizer1)
        self.Centre()


        
class LoginDialog_5(wx.Dialog):
    def __init__(self, parent, id):
        super(LoginDialog_5, self).__init__(parent, id, u'Information error',size=(200,200))
        self.sizer1 = wx.BoxSizer(wx.VERTICAL)
        self.sizer1.Add(wx.StaticText(self, -1, u"Invalid random forest parameters"),
                         0, wx.ALIGN_CENTER|wx.TOP, border=50)
        self.sizer1.Add(wx.Button(self, wx.ID_OK), 0, wx.ALIGN_CENTER_HORIZONTAL|wx.TOP|wx.BOTTOM, border=10 )
        self.SetSizer(self.sizer1)
        self.Centre()


class LoginDialog_6(wx.Dialog):
    def __init__(self, parent, id):
        super(LoginDialog_6, self).__init__(parent, id, u'Information error',size=(300,200))
        self.sizer1 = wx.BoxSizer(wx.VERTICAL)
        self.sizer1.Add(wx.StaticText(self, -1, u"Missing or invalid support vector machine parameters"),
                         0, wx.ALIGN_CENTER|wx.TOP, border=50)
        self.sizer1.Add(wx.Button(self, wx.ID_OK), 0, wx.ALIGN_CENTER_HORIZONTAL|wx.TOP|wx.BOTTOM, border=10 )
        self.SetSizer(self.sizer1)
        self.Centre()

        
class LoginDialog_7(wx.Dialog):
    def __init__(self, parent, id):
        super(LoginDialog_7, self).__init__(parent, id, u'File error',size=(400,200))
        self.sizer1 = wx.BoxSizer(wx.VERTICAL)
        self.sizer1.Add(wx.StaticText(self, -1, u"Please enter the correct test data file"),
                         0, wx.ALIGN_CENTER|wx.TOP, border=50)
        self.sizer1.Add(wx.Button(self, wx.ID_OK), 0, wx.ALIGN_CENTER_HORIZONTAL|wx.TOP|wx.BOTTOM, border=10 )
        self.SetSizer(self.sizer1)
        self.Centre()

