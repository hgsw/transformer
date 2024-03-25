**Transformer架构代码实现**

实现Transformer架构中的各个部分，最后搭建所有子模块，在一个copy任务中验证是模型是否具备最基础的学习能力。

**项目结构**

- embedding 词嵌入以及位置编码实现。
- encoder 编码层，主要包含多头注意力模块、前馈全连接等。
- decoder 解码层，主要包含两个多头注意力、前馈全连接等。
- output 输出层，主要是一个全连接和softmax层。
- model 模型搭建，将以上模块按transformer流程图搭建。
- utils 通用工具包，主要有注意力机制实现、clone函数、通用规范化层、数据生成函数、连接结构等。
- main 程序主入口，同时定义了损失函数、优化器、和简单的测试代码。


