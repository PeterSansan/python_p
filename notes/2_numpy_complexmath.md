
# numpy教程2_高级数学

## 1.线性代数模型(linalg)


```python
import numpy as np
```


```python
a = np.array([3,4])
np.linalg.norm(a) # 求绝对值
```




    5.0




```python
b = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
c= np.array([1,0,1])
# 矩阵和向量之间的算法
```


```python
np.dot(b,c)
```




    array([ 4, 10, 16])




```python
np.dot(c,b.T)
```




    array([ 4, 10, 16])




```python
np.trace(b)  # 求矩阵的迹
```




    15




```python
np.linalg.det(b) # 行列式值
```




    0.0




```python
np.linalg.matrix_rank(b)  # 矩阵的秩，2，不满秩，因为行与行之间等差
```




    2




```python
d = np.array([
    [2,1],
    [1,2]
])
```


```python
u,v = np.linalg.eig(d) 
print u,v   # 特征值与特征向量
```

    [ 3.  1.] [[ 0.70710678 -0.70710678]
     [ 0.70710678  0.70710678]]



```python
u,v = np.linalg.eigh(d)  # eigh更快更稳定，不过输出的值的顺序和eig（）是相反的
print u,v
```

    [ 1.  3.] [[-0.70710678  0.70710678]
     [ 0.70710678  0.70710678]]



```python
l = np.linalg.cholesky(d) # 分解为A*AT
```


```python
np.dot(l,l.T)
```




    array([[ 2.,  1.],
           [ 1.,  2.]])




```python
# c对非奇异矩阵，进行SVD分解并重建
e = np.array([
    [1,2],
    [3,4]])
```


```python
U,s,V = np.linalg.svd(e)
print "U =",U,"s =",s,"V =",V
```

    U = [[-0.40455358 -0.9145143 ]
     [-0.9145143   0.40455358]] s = [ 5.4649857   0.36596619] V = [[-0.57604844 -0.81741556]
     [ 0.81741556 -0.57604844]]



```python
np.dot(U,np.dot(s,V))
```




    array([ 5.43063125,  0.7129125 ])



## 2.随机模块(random)


```python
import numpy.random as random
```


```python
random.seed(42) #设置随机数种子
```


```python
random.rand(1,3) #　[0,1]之间
```




    array([[ 0.37454012,  0.95071431,  0.73199394]])




```python
random.random() # 产生一个[0,1]之间的浮点型随机数
```




    0.5986584841970366




```python
random.random([3,3]) # 下面几个的作用是一样的
```




    array([[ 0.15601864,  0.15599452,  0.05808361],
           [ 0.86617615,  0.60111501,  0.70807258],
           [ 0.02058449,  0.96990985,  0.83244264]])




```python
random.sample([3,3])
```




    array([[ 0.21233911,  0.18182497,  0.18340451],
           [ 0.30424224,  0.52475643,  0.43194502],
           [ 0.29122914,  0.61185289,  0.13949386]])




```python
random.random_sample([3,3])
```




    array([[ 0.29214465,  0.36636184,  0.45606998],
           [ 0.78517596,  0.19967378,  0.51423444],
           [ 0.59241457,  0.04645041,  0.60754485]])




```python
random.ranf([3,3])
```




    array([[ 0.17052412,  0.06505159,  0.94888554],
           [ 0.96563203,  0.80839735,  0.30461377],
           [ 0.09767211,  0.68423303,  0.44015249]])




```python
# 产生10个[1,6]之间的浮点型随机数
5*random.random(10)+1
```




    array([ 5.84792314,  4.87566412,  5.69749471,  5.47413675,  3.98949989,
            5.60937118,  1.44246251,  1.97991431,  1.22613644,  2.62665165])




```python
random.uniform(1,6,10)
```




    array([ 2.94338645,  2.35674516,  5.14368755,  2.78376663,  2.40467255,
            3.71348042,  1.70462112,  5.0109849 ,  1.37275322,  5.93443468])




```python
# 产生10个[1,6]之间的整型型随机数
random.randint(1,6,10)
```




    array([1, 4, 3, 3, 1, 3, 3, 1, 3, 5])




```python
# 产生2*5的标准正态分布样本
random.normal(size=(5,2))
```




    array([[ 0.21158701,  0.59704465],
           [-0.89633518, -0.11198782],
           [ 1.46894129, -1.12389833],
           [ 0.9500054 ,  1.72651647],
           [ 0.45788508, -1.68428738]])




```python
# 产生5个,n=5,p=0.5的二项分布样本 
random.binomial(n=5,p=0.5,size=5)
```




    array([2, 5, 4, 3, 2])




```python
a = np.arange(10)
```


```python
# 从a中随机采样7个
random.choice(a,7)
```




    array([6, 8, 9, 9, 2, 6, 0])




```python
# 从a中无回放的随机采样7个
random.choice(a,7,replace=False)
```




    array([7, 1, 8, 0, 2, 6, 4])




```python
# 对a进行乱序并返回一个新的array
random.permutation(a)
```




    array([8, 9, 4, 0, 7, 1, 3, 5, 2, 6])




```python
# 对a进行打散操作
random.shuffle(a)
a
```




    array([9, 4, 1, 0, 6, 3, 2, 5, 8, 7])




```python
# 生成一个长充为9的随机bytes序列并作秋str返回
random.bytes(9)
```




    '\xe0|C\xb12\x7f\xaf\x1e\xe9'



随机模块可以很方便地让我们做一些快速模拟去验证一些结论。比如来考虑一个非常违反直觉的概率题例子：一个选手去参加一个TV秀，有三扇门，其中一扇门后有奖品，这扇门只有主持人知道。选手先随机选一扇门，但并不打开，主持人看到后，会打开其余两扇门中没有奖品的一扇门。然后，主持人问选手，是否要改变一开始的选择？

这个问题的答案是应该改变一开始的选择。在第一次选择的时候，选错的概率是2/3，选对的概率是1/3。第一次选择之后，主持人相当于帮忙剔除了一个错误答案，所以如果一开始选的是错的，这时候换掉就选对了；而如果一开始就选对，则这时候换掉就错了。根据以上，一开始选错的概率就是换掉之后选对的概率（2/3），这个概率大于一开始就选对的概率（1/3），所以应该换。虽然道理上是这样，但是还是有些绕，要是通过推理就是搞不明白怎么办，没关系，用随机模拟就可以轻松得到答案：


```python
import numpy.random as random

random.seed(42)

# 做10000次实验
n_tests = 10000

# 生成每次实验的奖品所在的门的编号
# 0表示第一扇门，1表示第二扇门，2表示第三扇门
winning_doors = random.randint(0, 3, n_tests)

# 记录如果换门的中奖次数
change_mind_wins = 0

# 记录如果坚持的中奖次数
insist_wins = 0

# winning_door就是获胜门的编号
for winning_door in winning_doors:

    # 随机挑了一扇门
    first_try = random.randint(0, 3)
    
    # 其他门的编号
    remaining_choices = [i for i in range(3) if i != first_try]
  
    # 没有奖品的门的编号，这个信息只有主持人知道
    wrong_choices = [i for i in range(3) if i != winning_door]

    # 一开始选择的门主持人没法打开，所以从主持人可以打开的门中剔除
    if first_try in wrong_choices:
        wrong_choices.remove(first_try)
    
    # 这时wrong_choices变量就是主持人可以打开的门的编号
    # 注意此时如果一开始选择正确，则可以打开的门是两扇，主持人随便开一扇门
    # 如果一开始选到了空门，则主持人只能打开剩下一扇空门
    screened_out = random.choice(wrong_choices)
    remaining_choices.remove(screened_out)
    
    # 所以虽然代码写了好些行，如果策略固定的话，
    # 改变主意的获胜概率就是一开始选错的概率，是2/3
    # 而坚持选择的获胜概率就是一开始就选对的概率，是1/3
    
    # 现在除了一开始选择的编号，和主持人帮助剔除的错误编号，只剩下一扇门
    # 如果要改变注意则这扇门就是最终的选择
    changed_mind_try = remaining_choices[0]

    # 结果揭晓，记录下来
    change_mind_wins += 1 if changed_mind_try == winning_door else 0
    insist_wins += 1 if first_try == winning_door else 0

# 输出10000次测试的最终结果，和推导的结果差不多：
# You win 6616 out of 10000 tests if you changed your mind
# You win 3384 out of 10000 tests if you insist on the initial choice
print(
    'You win {1} out of {0} tests if you changed your mind\n'
    'You win {2} out of {0} tests if you insist on the initial choice'.format(
        n_tests, change_mind_wins, insist_wins
        )
)
```

    You win 6616 out of 10000 tests if you changed your mind
    You win 3384 out of 10000 tests if you insist on the initial choice



```python

```
