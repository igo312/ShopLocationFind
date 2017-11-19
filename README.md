# 天池竞赛汇总

好久不见，最近在忙天池的竞赛，两个星期真的好累。

[链接在这里](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100150.711.3.2def20dfm5UTlH&raceId=231620)

在这里依旧感谢@麦芽的baseline，不管最终结果如何，没有他的开源程序我在算法学到的东西少之甚少

如下我将在这里罗列出我的整个比赛想法与代码作为这次比赛的总结

必须申明，因精力优先，所有的参数在我心里并没有调试好。

## 一.数据分析

我上传的[ipny文件](https://github.com/igo312/ShopLocationFind/blob/master/customer_data_analyse.ipynb)其实十分简单，并且这是我最早期用来可视化的用途,其中含有数据集的形式可供参考

对于数据分析我主要做了三点

  ### [1.对于wifi的筛选](https://github.com/igo312/ShopLocationFind/tree/master/wifi_process)

  wifi具有强度，名称，是否连接的属性。第一我获取了频次，对于频次过于低的wifi，我认为是不重要的所以除去。

  第二我保留强度前2的wifi，并且对于连接上的wifi的强度进行了重新赋值。

  第三我考虑了公共wifi，即高于多少频次认为是公共wifi（但是在计算时我并没有删除）

  ### [2.对于时间的处理](https://github.com/igo312/ShopLocationFind/tree/master/time_process)

  我变换了时间为周几，小时，分钟

  但是在训练的过程中，我发现没有很大的影响

  又出于计算成本的考虑，并没有加入时间

  同时，时间的抽取是按分钟的，出现了同一时间同一店铺同一用户不同位置的情况。认为是不利于训练故放弃
  
   ### 3.对于位置的处理

  我求取了店铺经纬度与用户经纬度之间的真实距离，发现含有距离过远的用户，即异常值

  那么我的做法是对于这些异常值重设了经纬度，其赋值为店铺的经纬度并加上一点点距离

  同时我发现某些店铺对应的用户只有一个，这可能导致数据的不平衡，但我没有解决


## 二.模型训练
  
  [模型训练](https://github.com/igo312/ShopLocationFind/tree/master/model)

  在获取保留wifi或者经纬度距离保留值参数我使用了lightgbm

  但是因为对于lightgbm的调參没有找到相应的资料

  在模型训练中选择了xgboost

  但是我认为并没有做到对于xgboost百分百的优化

  同时我将用于验证集的early_stop用于了训练，因为了多个样本，我并不知道这是否合适

  其中一个问题是训练是针对于商场展开的，一个商场一个模型

  这样对于机器的硬件要求降低，但不一定是一个好模型

