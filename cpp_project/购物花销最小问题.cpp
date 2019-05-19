#include "iostream"
#include "algorithm"
#include "cmath"
#include "fstream"
using namespace std;
#define MAXB 6

/*
cost(a,b,c,d,e)表示购买商品组合(a,b,c,d,e)所需的最少费用
A[K], B[K], C[K], D[K], E[K]表示第K种商品优惠组合方案
offer(K)是第K种优惠组合方案的价格
coat(a,b,c,d,e) = cost(a-A[K], b-B[K], c-C[K], d-D[K], e-E[K]) + offer(K)
*/


int cost[MAXB][MAXB][MAXB][MAXB][MAXB];  //每种商品不超过5件
int product[MAXB];  //product[i]表示第i种商品件数
int B;  //需购商品的种数
int S;  //优惠方案的种数
int num[MAXB];  //num[i]表示编号为i的商品的新编号，范围为[1,5]
/*
purch[i][0]是第i种需购商品的件数
purch[i][1]是第i中需购商品的单价
*/
int purch[MAXB][2];

/*
offer[i][0]是第i种优惠组合的价格
offer[i][j]是第i种优惠组合中第j种商品的件数
*/
int offer[101][MAXB];  //最多有100种优惠组合

void mincost()
{
    int a, b, c, d, e, i;
    int minm = 0;
    for(i=1; i<=B; i++)  //以单价买最贵
        minm += product[i] * purch[i][1];

    for(i=1; i<=S; i++)  //优惠方案种数
    {
        a = product[1] - offer[i][1];  //还需要买的商品数量
        b = product[2] - offer[i][2];
        c = product[3] - offer[i][3];
        d = product[4] - offer[i][4];
        e = product[5] - offer[i][5];
        if(a>=0 && b>=0 && c>=0 && d>=0 && e>=0 &&
            cost[a][b][c][d][e]+offer[i][0]<minm)   //更新最优值
            minm = cost[a][b][c][d][e]+offer[i][0];
    }
    cost[product[1]][product[2]][product[3]][product[4]][product[5]] = minm;
}

void dyna(int i)  //迭代计算！
{
    if(i>B) //说明B种商品的product[]已经确定
    {
        mincost();
        return;
    }
    int j;
    for(j=0; j<=purch[i][0]; j++)
    {
        product[i] = j;  //第i种商品取值范围为[0,purch[i]]
        dyna(i+1);  //第i+1种商品
    }
}

int main()
{
    ifstream fin("购物.txt");

    //初始化
    memset(offer, 0, sizeof(offer));
    memset(product, 0, sizeof(product));
    memset(purch, 0, sizeof(purch));

    //输入数据
    cout << "需购商品的种数：";
    fin >> B; cout << B;
    cout << "\n输入需购各种商品的编号、件数、单价：\n";
    int i;
    int code;
    for(i=1; i<=B; i++)
    {
        fin >> code >> purch[i][0] >> purch[i][1];
        num[code] = i;  //用作重新编号
        cout << code << " " << purch[i][0] << " " << purch[i][1] << endl;
    }
    cout << "输入优惠方案种数：" ;
    fin >> S;  cout << S;
    cout << "\n输入各种优惠方案中商品种数、编号、件数、价格\n";
    int n, j, p;
    for(i=1; i<=S; i++)
    {
        fin >> n; cout << n << " ";
        for(j=1; j<=n; j++)
        {
            fin >> code >> p;
            cout << code << " " << p <<  " ";
            offer[i][num[code]] = p;
        }
        fin >>  offer[i][0];  cout << offer[i][0];
        cout << endl;
    }

    //输出结果
    dyna(1);
    cout << "\n最少购物费用为：" << cost[purch[1][0]][purch[2][0]][purch[3][0]][purch[4][0]][purch[5][0]] << endl;
    fin.close();
    return 0;
}
