# -*- coding: utf-8 -*-
import wx
from Panel.panel_1 import pssm_panel

class ALL_NoteBook(wx.Notebook):
    def __init__(self, parent, id):
        wx.Notebook.__init__(self, parent, id)
        
        self.panels = []
        
        ############################################################
        self.pssmPanel = pssm_panel(self, -1)
        self.panels.append(self.pssmPanel)
        self.AddPage(self.panels[0], u"Loess Collapsibility Coefficient Prediction")
        ############################################################
