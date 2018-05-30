# KDDCup 2018

[kdd_2018](https://biendata.com/competition/kdd_2018/)

## 数据

[北京2017年天气数据](https://pan.baidu.com/s/1YIsC4fXWoNk5vceQtzSTvA) 提取码：z8jp

[北京2017年空气质量数据](https://pan.baidu.com/s/1hWW-q_qQoAaXEhk18w4Syw) 提取码：i4kw 

[北京监测点数据](https://pan.baidu.com/s/1ynBrulxLkrtyJSBOggZ7Qw) 提取码：eykr 

[北京2017年网格天气数据](https://pan.baidu.com/s/13orE2RP-VKBLFzpSu_sJXA) 提取码：p4rp

[伦敦天气数据](https://pan.baidu.com/s/1aw0eTmBBl65VpF2rmVQqOw) 提取码：ig8c 

[伦敦预测站点空气质量数据](https://pan.baidu.com/s/1-BdbA9hPhQJLTH3ExV5VZQ) 提取码：gj97 

[伦敦其他站点空气质量数据](https://pan.baidu.com/s/15NTV7lYTw3Q3DWk7GkuYYg) 提取码：pa44 

[伦敦监测点数据](https://pan.baidu.com/s/12xLg-DzSHaguiuLwKqLCoQ) 提取码：c2kw

## 解决方案

- 数据清洗

数据清洗主要针对提供的数据中的缺失值和不正常数据。对于某一天除站点位置信息外所有空气质量信息全部缺失的一行数据直接舍弃，只有部分数据缺失尝试进行填充。尝试过的填充方案有用0填充，用均值填充和用前驱数据填充，从结果来看，填充的数据不应该破坏随时间平滑变化的趋势，因此采用前驱数据进行填充是一个不错的选择（预测方向为时间上的向后，所以不采用后继数据填充）。除了缺失值以外，数据中还有很多非正常数据。比如说明中提到的静风风速的表示，和明显过大的偏离数据，这部分数据采用在有效范围内随机生成的数据替代。在zhiwuyuan这个站点还出现了一行数据全部用999表示的脏数据，甚至在一段较长的连续时间内都缺少有效数据，如果需要使用这部分可以考虑采用距离较近的网格数据替代。为了便于观察数据的特征，对部分站点的数据连续变化情况进行了可视化处理。此外，需要预测的站点与提供的天气数据的地点并不匹配，需要手动选择合适的网格数据或外来数据进行填充。

- 预测模型

针对清洗后的数据进行预测模型的尝试。在尝试过程中，为了减少训练和测试时间，采用过去三小时预测未来一小时的简单任务来进行（合理性有待考察）。尝试的预测模型有线性回归（直接线性回归、ridge正则化、lasso正则化），SVR，xgboost和random forest，每次尝试都会对过去的空气质量数据和天气数据进行不同的取舍。测试结果显示，直接的线性回归在分数上较优。且从相关系数和对天气数据的取舍比较结果来看，完全舍弃天气数据而只用过去的空气质量数据进行预测的性能反而较优（然而通常情况下特征的选取对于预测结果的准确性应该有较大影响，出现这种现象可能是由于进行模型尝试时的预测跨度较短，仅从过去三小时预测未来一小时的数据，使得天气数据特征未能表现出其影响）。

在比赛正式提交阶段，需要预测未来48小时的空气质量数据。于是采用过去5天的空气质量数据和天气数据作为特征预测未来两天的空气质量数据。选择的模型主要是线性回归模型和seq2seq网络。两个模型的在预测结果上十分相似。通常的分数分布在0.38 ~ 0.55之间。用这两个模型形成了自动化获取数据并提交的脚本

- 说明

实际比赛时间以UTC时间为准，即北京时间的早上8:00对应UTC时间的0:00。因此北京时间早上8:00之前的提交实际上会被计算到上一天的提交中（前期因为没注意到这一点导致有10天左右的提交都错了一天）。来自清华大学计算机系本科生数据挖掘课堂的行之有效的模型有：Linear Regression、 MLP、 seq2seq、 xgboost、 LightGBM、 forecastxgb、 LSTM with CNN-RNN。



## 同类问题参考

#### KDDCup2017 车流量预测任务参考

[Rank13](https://github.com/InfiniteWing/KDD-2017-Travel-Time-Prediction)

[Rank11](https://github.com/lseiyjg/KDDCup2017_TravelTime)

[Rank??](https://github.com/Engineering-Course/kddcup2017)

#### Kaggle预测任务参考

[Kaggle房价预测Top15%](https://www.cnblogs.com/irenelin/p/7400388.html)

[Kaggle房价预测优胜方案](http://ju.outofmemory.cn/entry/309118)

[Kaggle网站流量预测优胜方案--基于RNN深度学习模型](https://blog.csdn.net/uwr44uouqcnsuqb60zk2/article/details/78794503)

[Kaggle空气质量预测优胜解决方案](https://github.com/benhamner/Air-Quality-Prediction-Hackathon-Winning-Model)