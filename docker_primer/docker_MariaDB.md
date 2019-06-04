 # 拉取镜像

> sudo docker pull mariadb 

# 使用镜像

> ```
> sudo docker run -p 3306:3306 --name mariaDB -v $PWD/conf:/etc/mysql/conf.d -v $PWD/logs:/logs -v $PWD/data:/var/lib/mysql -e MYSQL_ROOT_PASSWORD=123456 -d 
> ...
> ```

- -p 3306:3306：将容器的 3306 端口映射到主机的 3306 端口。
- -v -v $PWD/conf:/etc/mysql/conf.d：将主机当前目录下的 conf/my.cnf 挂载到容器的 /etc/mysql/my.cnf。
- -v $PWD/logs:/logs：将主机当前目录下的 logs 目录挂载到容器的 /logs。
- -v $PWD/data:/var/lib/mysql ：将主机当前目录下的data目录挂载到容器的 /var/lib/mysql 。
- -e MYSQL_ROOT_PASSWORD=123456：初始化 root 用户的密码。
- -d 后台运行容器，并返回容器id

初始化后：

开启：

> sudo docker start MariaDB

结束：

> sudo docker stop MariaDB

# 进入容器

> sudo docker exec -it mariaDB bash



# 退出

> exit
>
> or
>
> Ctrl+D

