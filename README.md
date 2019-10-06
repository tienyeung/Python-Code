# Git

## 步骤

第一步:首先,选择一个合适的地方,创建一个空目录

> $ cd ～
> $ mkdir mygithub
> $ cd mygithub

第二步:通过git init命令把这个目录变成Git可以管理的仓库

> $ git init

第三步 把文件添加到版本库

在./mygithub目录下 新建一个README.txt文件，并将其提交到缓存区

> $ git add README.txt

第四步 用命令git commit告诉Git,把文件提交到仓库

> $ git commit -m "hello github" 

*#-m 表示描述信息*

第五步:输入远程地址

> $ git remote add origin https:*//github.com/yeungtien/...*

*#origin 是默认远程仓库标识*

第六步:上传到github

> $ git push -u origin master



## 忽略文件：

通过**.gitignore**文件忽略你不想看到的文件

星号（*）匹配零个或多个任意字符；
[abc] 匹配任何一个列在方括号中的字符（这个例子要么匹配一个 a，要么匹配一个 b，要么匹配一个 c）；
问号（?）只匹配一个任意字符；
如果在方括号中使用短划线分隔两个字符，表示所有在这两个字符范围内的都可以匹配（比如 [0-9] 表示匹配所有 0 到 9 的数字）。

举个栗子：

#忽略所有 .a 结尾的文件

> *.a

#但 lib.a 除外

> !lib.a

#仅仅忽略项目根目录下的 TODO 文件

#不包括 subdir/TODO

> /TODO

#忽略 build/ 目录下的所有文件

> build/

## 删除文件：

> git rm to_be_deleted.txt

> git commit -m 'remove file'

## 撤销删除：

如果文件被删除：

> git checkout -- readme.txt

如果一个修改后的文件已经被暂存了，恢复到之前的状态：

> git reset HEAD readme.txt

如果文件修改已经被 commit 了，如何撤销：

> git commit --amend

## 版本回退：

工作目录中运行 git log

HEAD，它指向的是最新的提交。而上一次的提交就是 HEAD^，上上次是 HEAD^^，也可以写成 HEAD~2，以此类推。

要回退上一个版本，只要：

> git reset --hard HEAD^

或提交id：

> git reset --hard 15547（不必输全）

## 多地管理

1. 把项目clone到本地
2. 初始化项目为git 

> git init

 3.配置邮箱

> git config --global user.email "yeungtien@gmail.com"

3.1 配置origin

> git remote add origin https://github.com/tienyeung/leetcode

4.1 生成ssh密钥并添加到setting(用于ssh clone)

> ssh-keygen -t rsa -C 'yeungtien@gmail.com'

将生成的id_rsa.pub中的密钥添加到自己账户

4.2 提交无需密码(用于https clone)

> git config --global credential.helper store

4.3 更改协议

> git remote set-url origin git@github.com:yourusername/yourrepositoryname.git

## 常见问题

[解决github permission denied(publickey)问题](<https://www.jianshu.com/p/f22d02c7d943>)

[git批量提交和删除](<https://blog.csdn.net/pan0755/article/details/78460149>)

[多地管理github账号](https://blog.csdn.net/xingkong_hdc/article/details/79484518)

## 参考

[Github 入门](<https://www.jianshu.com/p/38611735b15e>)