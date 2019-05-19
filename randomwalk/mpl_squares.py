import matplotlib.pyplot as plt

x_value=list(range(1,1001))
y_value=[x**2 for x in x_value]
plt.scatter(x_value,y_value,c=(1,0,0.8),edgecolor='none',s=30)
#设置图标属性
plt.title('square',fontsize=24)
plt.xlabel('value',fontsize=14)
plt.ylabel('square of value',fontsize=14)
plt.axis([0,1100,0,1100000])

#自动保存
plt.savefig('plot_test',bbox_inches='tight')
