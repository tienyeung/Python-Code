# 我的第一个github仓库（2019/5/18）

从无到有，上手也倒快速。从一窍不通的git菜鸟，到现在大致了解其间的流程，也算正式迈入了开源世界的门槛。既然开了这个仓库，就有了一直维护到底的决心，虽然难说固定一段时间就能提交一些代码，但起码我会“笔耕不缀”得进行python项目的学习和实践。我一直相信学无止境，在IT行业尤为如此，知识更新换代速度太快了，曾经的state-of-art，过几个月就能被刷新记录。开源仓库于我而言，仿佛一个庞大的充满新知识和新视野的世界，里面有数不清的精妙绝伦或思维跳脱的项目或代码等着我去发现和实践，我对此充满了热情与敬意！

我学习python是为了什么，我也不是很清楚，为了工作？为了兴趣？可能两者兼顾，又或者仅仅是想让机器帮我完成一些重复或者难以短时间实现的任务，总之学习是有趣的，尤其是上手了一个小项目并且完成了平时难以完成的任务时，那种欣喜和激动，无法用言语形容：）

# Git

## 步骤

第一步:首先,选择一个合适的地方,创建一个空目录

$ cd ～
$ mkdir mygithub
$ cd mygithub

第二步:通过git init命令把这个目录变成Git可以管理的仓库

$ git init

第三步 把文件添加到版本库

在./mygithub目录下 新建一个README.txt文件，并将其提交到缓存区

$ git add README.txt

第四步 用命令git commit告诉Git,把文件提交到仓库

$ git commit -m "hello github" 

*#-m 表示描述信息*

第五步:输入远程地址

$ git remote add origin https:*//github.com/itmyhome2013/mygithub.git*

*#origin 是默认远程仓库标识*

第六步:上传到github

$ git push -u origin master



## 忽略文件：

通过**.gitignore**文件忽略你不想看到的文件

星号（*）匹配零个或多个任意字符；
[abc] 匹配任何一个列在方括号中的字符（这个例子要么匹配一个 a，要么匹配一个 b，要么匹配一个 c）；
问号（?）只匹配一个任意字符；
如果在方括号中使用短划线分隔两个字符，表示所有在这两个字符范围内的都可以匹配（比如 [0-9] 表示匹配所有 0 到 9 的数字）。

举个栗子：

#忽略所有 .a 结尾的文件

*.a

#但 lib.a 除外

!lib.a

#仅仅忽略项目根目录下的 TODO 文件

#不包括 subdir/TODO

/TODO

#忽略 build/ 目录下的所有文件

build/

## 删除文件：

git rm to_be_deleted.txt

git commit -m 'remove file'

## 撤销删除：

如果文件被删除：

git checkout -- readme.txt

如果一个修改后的文件已经被暂存了，恢复到之前的状态：

git reset HEAD readme.txt

如果文件修改已经被 commit 了，如何撤销：

git commit --amend

## 版本回退：

工作目录中运行 git log

HEAD，它指向的是最新的提交。而上一次的提交就是 HEAD^，上上次是 HEAD^^，也可以写成 HEAD~2，以此类推。

要回退上一个版本，只要：

git reset --hard HEAD^

或提交id：

git reset --hard 15547（不必输全）

## 常见问题

[解决github permission denied(publickey)问题](<https://www.jianshu.com/p/f22d02c7d943>)

[git批量提交和删除](<https://blog.csdn.net/pan0755/article/details/78460149>)

## 参考

[Github 入门](<https://www.jianshu.com/p/38611735b15e>)