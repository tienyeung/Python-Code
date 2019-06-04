# WIKIDATA

- **数据提取**

- **分表索引**

- **图形化界面实现**

## 准备

- 根据问题设计表，要求：了解mysql数据库的**语法**（创建数据库，创建表，创建索引），wikidata json文件的**逻辑结构**（难点）

- 理解5个问题所指，比如entity，subclass of为何意等。。。

- pymysql库的使用

- tkinter库的使用

  

## 领会

- 庞大数据库的操作（50，000，000+数据）
- **sql语法**的使用
- wiki数据库原理的浅显认识（类似于知识图谱）
- 压缩包（BZ2格式）无需完全解压（若完全解压有700G+），可用BZ2File逐行读取
- tkinter的浅显认识



## 总结

这是一次较为开放的作业，需独自完成庞大数据的提取分表入库，查询语言的设计及图形化界面的实现。经此一役，受益匪浅。从对数据库的一无所知到逐渐了解数据库的原理及操作，果然实践出真知，书上看到的理论知识直到用到时才会有更深刻的理解。

这个作业是在前辈的基础上修改的，慢慢摸索不断修改从而逐渐完善，看似简单的项目依然花费了我相当的精力和时间，但就算到此刻这个项目仍有令我不满意的地方，比如问题的理解仍然不到位，wikijson的层次关系仍然不明了。

此项目有诸多遗漏/错误/不足/失败，若有幸被看到，请务必批评指正，我定不胜感激。

## TODO

或者叫改正的地方（因为我应该不会改了，懒- -||）

- 表中字段类型或范围的设计
- 界面的优化
- 索引的创建
- 。。。