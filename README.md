# 我的第一个github仓库（2019/5/18）

从无到有，上手也倒快速。从一窍不通的git菜鸟，到现在大致了解其间的流程，也算正式迈入了开源世界的门槛。既然开了这个仓库，就有了一直维护到底的决心，虽然难说固定一段时间就能提交一些代码，但起码我会“笔耕不缀”得进行python项目的学习和实践。我一直相信学无止境，在IT行业尤为如此，知识更新换代速度太快了，曾经的state-of-art，过几个月就能被刷新记录。开源仓库于我而言，仿佛一个庞大的充满新知识和新视野的世界，里面有数不清的精妙绝伦或思维跳脱的项目或代码等着我去发现和实践，我对此充满了热情于敬意！

我学习python是为了什么，我也不是很清楚，为了工作？为了兴趣？可能两者兼顾，又或者仅仅是想让机器帮我完成一些重复或者难以短时间实现的任务，总之学习是有趣的，尤其是上手了一个小项目并且完成了平时难以完成的任务时。

# Git

## 步骤

第一步:首先,选择一个合适的地方,创建一个空目录**

$ cd ～
$ mkdir mygithub
$ cd mygithub

**第二步:通过git init命令把这个目录变成Git可以管理的仓库**

$ git init

**第三步 把文件添加到版本库**

在./mygithub目录下 新建一个README.txt文件，并将其提交到缓存区

$ git add README.txt

**第四步 用命令git commit告诉Git,把文件提交到仓库**

$ git commit -m "hello github" 

*#-m 表示描述信息*

**第五步:输入远程地址**

$ git remote add origin https:*//github.com/itmyhome2013/mygithub.git*

*#origin 是默认远程仓库标识*

**第六步:上传到github**

$ git push -u origin master

## 常见问题

[解决github permission denied(publickey)问题](<https://www.jianshu.com/p/f22d02c7d943>)

[git批量提交和删除](<https://blog.csdn.net/pan0755/article/details/78460149>)

## 参考

[Github 入门](<https://www.jianshu.com/p/38611735b15e>)

