# 贡献代码


## 1. 代码结构分析（持续更新中）

整体代码结构如下所示。


```
./AnomalyDetection          # 项目目录根目录
|-- configs               # 配置文件目录
|-- docs                  # 文档目录，包括模型库、使用教程等
|-- deploy                # 预测部署相关
|   ├── py_inference      # 基于PaddleInference的python推理代码目录
|   ├── py_serving        # 基于PaddleServing的python推理代码目录
|   ├── cpp_inference     # 基于PaddleInference的cpp推理代码目录
|   ├── cpp_serving       # 基于PaddleServing的cpp推理代码目录
|-- tools                 # 工具类代码目录
|   ├── train.py          # 训练代码文件
|   ├── eval.py           # 评估代码文件
|   ├── predict.py        # 基于动态图模型的推理代码文件
|   ├── export.py         # 模型导出代码文件
|-- ppad                  # 核心代码目录
|   |-- data              # 数据处理与加载代码目录
|   |-- loss              # 损失函数定义目录
|   |-- metrics           # 指标计算代码目录
|   |-- modeling          # 模型组网目录
|   |-- optimizer         # 优化器、学习率、正则化方法定义目录
|   |-- postprocess       # 后处理方法目录
|   `-- utils             # 工具类目录，如日志、模型保存与加载、配置加载等
|-- test_tipc             # 训推一体测试目录
|-- README_en.md          # 英文用户手册
|-- README.md             # 中文用户手册
|-- LICENSE               # LICENSE文件
```


所有的模块（`ppad目录下面的内容`）需要基于类进行实现。

* 数据加载过程中，建议dataloader输出的batch为dict格式，方便不同的算法使用。一些基础的数据预处理类建议整理为可以组合的形式（类似于`Compose`的功能）。
* 模型组网过程中，建议按照模块进行区分，如backbone、neck、head等，保证子模块可配置。
* 优化器相关代码已经初步整理完成，在实现过程中可以进一步review，确认是否合理。
* 计算指标时，建议其实现基于`paddle.metric.Metric`类，保证统一性。


## 2. 提交代码

python代码、文档、提交pull request等规范请参考：[代码与文档规范](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/doc/doc_ch/code_and_doc.md)。
