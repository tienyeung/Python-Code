import tkinter as tk
from tkinter import *
from tkinter import scrolledtext
import pymysql
import time

conn = pymysql.connect(
    host='localhost',  # mysql服务器地址
    port=3306,  # 端口号
    user='root',  # 用户名
    passwd='12345678',  # 密码
    db='wikidataplus',  # 数据库名称
    charset='utf8mb4'  # 连接编码
)
cur = conn.cursor()  # 创建并返回游标


app = Tk()
app.title("WIKIDATA")
app.geometry("1050x565")


class Application(Frame):

    def __init__(app, master):
        super(Application, app).__init__(master)
        app.grid()
        app.create_widgets()

    def create_widgets(app):

        app.label1 = Label(app, text="Please input your query:")
        app.label1.grid(row=0, column=0, columnspan=10,
                        sticky=W, padx=5, pady=5)

        app.var1 = StringVar()
        app.var1.set('')
        app.e1 = Entry(app, width=55, textvariable=app.var1).grid(
            row=0, column=1, columnspan=5, ipady=1)

        app.label2 = Label(app, text="Please choose your query method:")
        app.label2.grid(row=1, column=0, columnspan=2,
                        sticky=W, padx=5, pady=5)

        app.bttn = Button(app, text="name->entity", command=app.cmd1)
        app.bttn.grid(row=1, column=2, ipadx=3, ipady=1,
                      sticky=W, padx=10, pady=10)


        app.bttn2 = Button(app, text="id->pre-category", command=app.cmd2)
        app.bttn2.grid(row=1, column=3, ipadx=3, ipady=1,
                       sticky=W, padx=10, pady=10)

        app.bttn3 = Button(app, text="id->all-entity", command=app.cmd3)
        app.bttn3.grid(row=1, column=4, ipadx=3, ipady=1,
                       sticky=W, padx=10, pady=10)

        app.bttn4 = Button(app, text="id->all-property", command=app.cmd4)
        app.bttn4.grid(row=1, column=5, ipadx=3, ipady=1,
                       sticky=W, padx=10, pady=10)

        app.bttn5 = Button(app, text="Q&A", command=app.cmd5)
        app.bttn5.grid(row=1, column=7, ipadx=1, ipady=1,
                       sticky=W, padx=10, pady=10)

        # app.label3 = Label(app, text="Optimize(INDEX) :")
        # app.label3.grid(row=2, column=0, columnspan=10,
        #                 sticky=W, padx=5, pady=5)

        # app.bttn6 = Button(app, text="name->entity", command=app.cmd6)
        # app.bttn6.grid(row=2, column=2, ipadx=3, ipady=1,
        #                sticky=W, padx=10, pady=10)

        # app.bttn7 = Button(app, text="id->pre-category", command=app.cmd7)
        # app.bttn7.grid(row=2, column=3, ipadx=3,
        #                ipady=1, sticky=W, padx=5, pady=5)

        # app.bttn8 = Button(app, text="id->all-entity", command=app.cmd8)
        # app.bttn8.grid(row=2, column=4, ipadx=3, ipady=1,
        #                sticky=W, padx=10, pady=10)

        # app.bttn9 = Button(app, text="id->all-property", command=app.cmd9)
        # app.bttn9.grid(row=2, column=5, ipadx=3, ipady=1,
        #                sticky=W, padx=10, pady=10)

        # app.bttn0 = Button(app, text="Q&A", command=app.cmd10)
        # app.bttn0.grid(row=2, column=7, ipadx=1, ipady=1,
        #                sticky=W, padx=10, pady=10)

        app.label4 = Label(app, text="Result:")
        app.label4.grid(row=3, column=0, columnspan=8,
                        sticky=W, padx=5, pady=5)

        scrolW = 100  # 设置文本框的长度
        scrolH = 25  # 设置文本框的高度
        app.scr = scrolledtext.ScrolledText(
            app, width=scrolW, height=scrolH, wrap=WORD)
        app.scr.grid(row=4, column=0, columnspan=8, sticky=W, padx=10, pady=10)
        app.scr.delete(0.0, END)
        app.quote = '   '
        app.scr.insert(END, app.quote)

        app.label4 = Label(app, text="Run Time/s:")
        app.label4.grid(row=3, column=10, columnspan=10,
                        sticky=W, padx=1, pady=1)

        scrolW = 10  # 设置文本框的长度
        scrolH = 5  # 设置文本框的高度
        app.scr1 = scrolledtext.ScrolledText(
            app, width=scrolW, height=scrolH, wrap=WORD)
        app.scr1.grid(row=4, column=12, columnspan=5, sticky=N, padx=1, pady=1)
        app.scr1.delete(0.0, END)
        app.quote = '   '
        app.scr1.insert(END, app.quote)

    def get_input(app):  # 获取输入框的查询内容
        return app.var1.get()


# 1)   Given a name, return all the entities that match the name.


    def cmd1(app):
        sql_index= 'alter table entity add index label_index (qlabel(50))'
        cur.execute(sql_index)
        start = time.clock()
        input1 = app.var1.get()
        sql = 'select * from entity where qlabel like "%%%%%s%%%%%%"' % (
            input1)
        cur.execute(sql)
        rows = cur.fetchall()
        # print (rows)
        app.scr.delete(0.0, END)
        app.scr1.delete(0.0, END)
        for row in rows:
            id = row[0]
            type = row[1]
            label = row[2]
            aliase = row[3]
            description = row[4]
            result = 'id:'+id.ljust(12)+'     '+'type:'+type.ljust(8)+'     '+'label:'+label.ljust(
                8)+'     '+'aliase:'+aliase.ljust(8)+'     '+'description:'+description.ljust(8)+'\n'
            app.quote = result
            app.scr.insert(END, app.quote)
        end = time.clock()
        app.time = end-start
        app.scr1.insert(END, app.time)


# 2) Given an entity, return all preceding categories (instance of and subclass of) it belongs to

    def cmd2(app):
        sql_index_1= 'create index qid_claim_index on claim(qid)'
        cur.execute(sql_index_1)
        sql_index_2= 'create index pid_property_index on property(pid)'
        cur.execute(sql_index_2)
        start = time.clock()
        input1 = app.var1.get()
        sql = 'select pid,plabel from property where plabel="subclass of" or "instance of" in (select plabel from property where pid in (select pid from claim where qid ="%s")) ' % (
            input1)
        cur.execute(sql)
        rows = cur.fetchall()
        app.scr.delete(0.0, END)
        app.scr1.delete(0.0, END)
        print(rows)
        for row in rows:
            # result = '%s' + 'instance of' + 'property:'+row[0]+'\n' % input1
            result = row[1]+' '+'property:'+row[0]+'\n' 
            app.quote = result
            app.scr.insert(END, app.quote)
        end = time.clock()
        app.time = end-start
        app.scr1.insert(END, app.time)


# 3)  Given an entity, return all entities that are co-occurred with this entity in one statement.


    def cmd3(app):
        start = time.clock()
        input1 = app.var1.get()
        sql = 'select datavalue_value from claim where qid ="%s" and datavalue_type="string"' % (
            input1)
        cur.execute(sql)
        rows = cur.fetchall()
        #print (rows)
        app.scr.delete(0.0, END)
        app.scr1.delete(0.0, END)
        for row in rows:
            # print(row[0])
            result = 'entityid:'+row[0]+'\n'
            app.quote = result
            app.scr.insert(END, app.quote)
        end = time.clock()
        #print("cost time:")
        app.time = end-start
        app.scr1.insert(END, app.time)


# 4)   Given an entity, return all the properties and statements it possesses.


    def cmd4(app):
        start = time.clock()
        input1 = app.var1.get()
        sql = 'select pid from claim where qid="%s" ' % (input1)
        cur.execute(sql)
        rows = cur.fetchall()
        #print (rows)
        app.scr.delete(0.0, END)
        app.scr1.delete(0.0, END)
        for row in rows:
            # print(row[0])
            result = 'property:' + \
                row[0].ljust(20)+'\n'
            app.quote = result
            app.scr.insert(END, app.quote)
        end = time.clock()
    #print("cost time:")
        app.time = end-start
        app.scr1.insert(END, app.time)

# 5）  提问与回答

    def cmd5(app):
        start = time.clock()
        input1 = app.var1.get()
        sql = 'select datavalue_value from claim where pid in (select pid from property where plabel like "%%%%%s%%%%%%") and qid in (select qid from entity where qlabel like "%%%%%s%%%%%%")' % (
            input1)
        cur.execute(sql)
        rows = cur.fetchall()
    #print (rows)
        app.scr.delete(0.0, END)
        app.scr1.delete(0.0, END)
        for row in rows:
            # print(row[0])
            result = 'value:'+row[0]+'\n'
            app.quote = result
            app.scr.insert(END, app.quote)
        end = time.clock()
    #print("cost time:")
        app.time = end-start
        app.scr1.insert(END, app.time)


#index
    # def cmd6(app):
    #     # sql_index= 'alter table entity add index label_index (qlabel(88))'
    #     # cur.execute(sql_index)
    #     start = time.clock()
    #     input1 = app.var1.get()
    #     sql = 'select * from entity where qlabel like "%%%%%s%%%%%%"' % (
    #         input1)
    #     cur.execute(sql)
    #     rows = cur.fetchall()
    #     # print (rows)
    #     app.scr.delete(0.0, END)
    #     app.scr1.delete(0.0, END)
    #     for row in rows:
    #         id = row[0]
    #         type = row[1]
    #         label = row[2]
    #         aliase = row[3]
    #         description = row[4]
    #         result = 'id:'+id.ljust(12)+'     '+'type:'+type.ljust(8)+'     '+'label:'+label.ljust(
    #             8)+'     '+'aliase:'+aliase.ljust(8)+'     '+'description:'+description.ljust(8)+'\n'
    #         app.quote = result
    #         app.scr.insert(END, app.quote)
    #     end = time.clock()
    #     app.time = end-start
    #     app.scr1.insert(END, app.time)

    # def cmd7(app):
    #     sql_index_1= 'create index qid_claim_index on claim(qid)'
    #     cur.execute(sql_index_1)
    #     sql_index_2= 'create index pid_property_index on property(pid)'
    #     cur.execute(sql_index_2)
    #     start = time.clock()
    #     input1 = app.var1.get()
    #     sql = 'select pid,plabel from property where plabel="subclass of" or "instance of" in (select plabel from property where pid in (select pid from claim where qid ="%s")) ' % (
    #         input1)
    #     cur.execute(sql)
    #     rows = cur.fetchall()
    #     app.scr.delete(0.0, END)
    #     app.scr1.delete(0.0, END)
    #     print(rows)
    #     for row in rows:
    #         # result = '%s' + 'instance of' + 'property:'+row[0]+'\n' % input1
    #         result = row[1]+' '+'property:'+row[0]+'\n' 
    #         app.quote = result
    #         app.scr.insert(END, app.quote)
    #     end = time.clock()
    #     app.time = end-start
    #     app.scr1.insert(END, app.time)

    # def cmd8(app):
    #     sql_index_1= 'create index type_claim_index on claim(datavalue_type)'
    #     cur.execute(sql_index_1)
    #     start = time.clock()
    #     input1 = app.var1.get()
    #     sql = 'select datavalue_value from claim where qid ="%s" and datavalue_type="string"' % (
    #         input1)
    #     cur.execute(sql)
    #     rows = cur.fetchall()
    #     #print (rows)
    #     app.scr.delete(0.0, END)
    #     app.scr1.delete(0.0, END)
    #     for row in rows:
    #         # print(row[0])
    #         result = 'entityid:'+row[0]+'\n'
    #         app.quote = result
    #         app.scr.insert(END, app.quote)
    #     end = time.clock()
    #     #print("cost time:")
    #     app.time = end-start
    #     app.scr1.insert(END, app.time)
    
    # def cmd9(app):
    #     start = time.clock()
    #     input1 = app.var1.get()
    #     sql = 'select pid from claim where qid="%s" ' % (input1)
    #     cur.execute(sql)
    #     rows = cur.fetchall()
    #     #print (rows)
    #     app.scr.delete(0.0, END)
    #     app.scr1.delete(0.0, END)
    #     for row in rows:
    #         # print(row[0])
    #         result = 'property:' + \
    #             row[0].ljust(20)+'\n'
    #         app.quote = result
    #         app.scr.insert(END, app.quote)
    #     end = time.clock()
    # #print("cost time:")
    #     app.time = end-start
    #     app.scr1.insert(END, app.time)

    # def cmd10(app):
    #     start = time.clock()
    #     input1 = app.var1.get()
    #     sql = 'select datavalue_value from claim where pid in (select pid from property where plabel like "%%%%%s%%%%%%") and qid in (select qid from entity where qlabel like "%%%%%s%%%%%%")' % (
    #         input1)
    #     cur.execute(sql)
    #     rows = cur.fetchall()
    # #print (rows)
    #     app.scr.delete(0.0, END)
    #     app.scr1.delete(0.0, END)
    #     for row in rows:
    #         # print(row[0])
    #         result = 'value:'+row[0]+'\n'
    #         app.quote = result
    #         app.scr.insert(END, app.quote)
    #     end = time.clock()
    # #print("cost time:")
    #     app.time = end-start
    #     app.scr1.insert(END, app.time)


if __name__ == '__main__':
    app = Application(app)
    app.get_input()
    app.mainloop()
