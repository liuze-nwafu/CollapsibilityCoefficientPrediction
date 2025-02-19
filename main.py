#!/usr/bin/env python
# -*- coding: utf-8 -*-
import wx
from Panel.all_notebook import ALL_NoteBook
import os,sys
#import xgboost
######################################################
Version = "1.0"
ReleaseDate = "2025-2-19"
######################################################

ID_EXIT = 200
ID_ABOUT = 204
ID_ABOUT1 = 205

class MainFrame(wx.Frame):
    def __init__(self, parent, id, title=''):
        wx.Frame.__init__(self, parent, id, title, pos=(0, 0), size=(1550, 800))
        ###########################################################################################
        ###########################################################################################

        ## 状态栏的创建
        self.setupStatusBar()

        ###########################################################################################
        ###########################################################################################
        ## 创建菜单栏
        self.setupMenuBar()

        ###########################################################################################
        ###########################################################################################
        ## 显示按钮功能
        self.initUI()

        self.Centre()

    def initUI(self):
        ###########################################################################################
        ###########################################################################################
        ## 显示按钮功能
        self.all_notebook = ALL_NoteBook(self, -1)

    def setupStatusBar(self):
        ###########################################################################################
        ###########################################################################################
        ## 状态栏的创建
        sb = self.CreateStatusBar(2)
        self.SetStatusWidths([-1, -2])
        self.SetStatusText("Ready", 0)
        
    def absolute_path(self,relative):
        if hasattr(sys,"_MEIPASS"):
            absolute_path1 = os.path.join(sys._MEIPASS,relative)
        else:
            absolute_path1 = os.path.join(relative)
        return absolute_path1

    def setupMenuBar(self):
        ###########################################################################################
        ###########################################################################################
        ## 创建菜单栏
        ###########################################################################################
        ## 主菜单
        menubar = wx.MenuBar()
        ###########################################################################################
        ## 子菜单 ：退出(Quit)
        fmenu = wx.Menu()
        quit_menu = wx.MenuItem(fmenu, ID_EXIT, u'Quit(&Q)')
        quit_menu.SetBitmap(wx.Bitmap(self.absolute_path('./image/quit.jpg')))  # 添加一个图标
        #quit_menu.SetBitmap(wx.Bitmap('E:\\tumu\image\quit.jpg'))  # 添加一个图标
        fmenu.Append(quit_menu)  # 将选项添加到菜单中
        ## 将子菜单添加到文件(File)中
        menubar.Append(fmenu, u'File(&F)')
        ###########################################################################################
        pmenu = wx.Menu()
        ###########################################################################################
        ###########################################################################################
        self.SetMenuBar(menubar)
        # 菜单中子菜单，事件行为的绑定即实现
        wx.EVT_MENU(self, ID_EXIT, self.OnMenuExit)
        wx.EVT_MENU(self, ID_ABOUT, self.OnMenuAbout)
        wx.EVT_MENU(self, ID_ABOUT1, self.About1)
        wx.EVT_CLOSE(self, self.OnCloseWindow)
        ###########################################################################################
        ###########################################################################################

    def OnMenuExit(self, event):
        self.Close()

    def OnMenuAbout(self, event):
        dlg = AboutDialog(None, -1)
        dlg.ShowModal()
        dlg.Destroy()

    def About1(self, event):
        dlg2 = About1Dialog(None, -1)
        dlg2.ShowModal()
        dlg2.Destroy()

    def OnCloseWindow(self, event):
        self.Destroy()

class About1Dialog(wx.Dialog):
    def __init__(self, parent, id):
        wx.Dialog.__init__(self, parent, id, 'Contact', size=(300, 150))

        self.sizer1 = wx.BoxSizer(wx.VERTICAL)
        self.sizer1.Add(wx.StaticText(self, -1, u"zeliu@nwafu.edu.cn"),
                        0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=20)
        self.sizer1.Add(wx.StaticText(self, -1, u"2718131423"),
                        0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=0)
        self.sizer1.Add(wx.Button(self, wx.ID_OK), 0, wx.ALIGN_CENTER | wx.BOTTOM, border=20)
        self.SetSizer(self.sizer1)
        self.Centre()


## 定义一个对话框
class AboutDialog(wx.Dialog):
    def __init__(self, parent, id):
        wx.Dialog.__init__(self, parent, id, 'About Me', size=(300, 250))

        self.sizer1 = wx.BoxSizer(wx.VERTICAL)
        self.sizer1.Add(wx.StaticText(self, -1, u"Loess Collapsibility Coefficient"),
                        0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=20)
        self.sizer1.Add(wx.StaticText(self, -1, u"Prediction software"),
                        0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=0)
        self.sizer1.Add(wx.StaticText(self, -1, u"(C) 2024 Yangling Vocational Technical College "),
                        0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=10)
        self.sizer1.Add(wx.StaticText(self, -1, "Version %s , %s" % (Version, ReleaseDate)),
                        0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=10)
        self.sizer1.Add(wx.StaticText(self, -1, u"Author : Zhang Wan"),
                        0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=10)
        self.sizer1.Add(wx.Button(self, wx.ID_OK), 0, wx.ALIGN_CENTER | wx.BOTTOM, border=20)
        self.SetSizer(self.sizer1)
        self.Centre()


class App(wx.App):
    def __init__(self):
        super(self.__class__, self).__init__()

    def OnInit(self):
        self.version = u""
        self.title = u"Loess Collapsibility Coefficient Prediction Software v1.0" + self.version
        frame = MainFrame(None, -1, self.title)
        frame.Show(True)

        return True


if __name__ == "__main__":
    app = App()
    app.MainLoop()
