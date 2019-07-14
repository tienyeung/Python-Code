/*最短路径问题*/
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
typedef long long ll;
using namespace std;
const int N=500;

struct node{
  int x;//节点编号
  ll d;//起点到节点长度
  node(int X,int D):x(X),d(D){}；
  bool operator < (const node &a) const//使用优先级队列，重载小于为大于,就可以对两个node的优先级进行比较
  {
    return d>a.d;
  }
}//表示节点编号和距离的结构体

int n,m;//路口数和道路数
ll dis[N],sum[N];//dis记录最小疲劳度;sum记录连续小路路径长度和
int f[N][N];//标记是大路还是小路
vector<node>r[N];

void input(){
  cin>>n>>m;
  for(int i=0;i<m;i++)
  {
    int flag,u,v;
    ll c;
    cin>>flag>>u>>v>>c;
    r[u].push_back(node(v,c));
    r[v].push_back(node(u,c));
    if(flag == 1)
       f[u][v]=f[v][u]=1；
  }
}//路径输入

void dij(){
  dis[1]=0;
  for(int i=2;i<=n;++i)
  {
    dis[i]=INF
  }//初始化dis[]
  priority_queue<node>q;//优先级队列q
  q.push(node（1，dis[1]));
  int vis[N]=0;
  while (!q.empty())
  {
    node e = q.top;
    q.pop();
    if(!vis[e.x])//若此节点没有访问到
    {
      vis[e.x]=1;
      for(int i=0;i<r[e.x].size();++1)//r[e.x].size()为与e相连的节点个数
      {
        node next=r[e.x][i];//下一个节点
        if(!f[e.x][next.x])//大路
        {
          if(dis[next.x]>dis[e.x]+next.d)
          {
            dis[next.x]=dis[e.x]+next.d;
            q.push(node(next.x,dis[next.x]));
            sum[next.x]=0;
          }
        }
        else//小路
        {
          if(!sum[e.x])//前一条是大路
          {
            if(dis[next.x]>dis[e.x]+next.d*next.d)
            {
              dis[next.x]=dis[e.x]+next.d*next.d;
              q.push(node(next.x,dis[next.x]));
              sum[next.x]=next.d;
            }
          }
          else//前一条是小路
          {
           if(dis[next.x]>((dis[e.x]+next.d)*(dis[e.x]+next.d))
           {
             dis[next.x]=(dis[e.x]+next.d])*(dis[e.x]+next.d);
             q.push(node(next.x,dis[next.x]));
             sum[next.x]=sum[e.x]+next.d
           }
          }
        }
      }
    }
  }
}//参照网上和算法书上迪杰斯特拉算法改编

int main(){
  input();
  dij();
  cout<<dis[n];
  return 0;
}
