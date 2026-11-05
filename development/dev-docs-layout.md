# 文档分类与扁平化整理

目的：按 paper2 的用途分类风格整理 paper1 文档，去掉重复归档路径、数字编号和过时源码副本。

范围：development、protocols、experiments、reports、decisions、resources，以及论文写作所需的 manuscript。规约集中到 protocols；每个实验最多一层有意义的实验 ID；文稿不携带源码快照。旧源码副本删除，必要数值输入与图表导出归属 results，冻结科学输入保持字节身份。

计划：先逐文件分类和命名，再移动文档、清理旧代码与别名，同步代码及图表对象引用，最后核查目录深度、文档链接和必要绘图入口。不开训练，不操作 R2，不改科学数值。

验收：没有 archive、数字前缀目录或旧代码文件；规约只有同类型入口；当前链接及冻结输入可读；受影响绘图流程通过。

执行结果：已按七类完成逐文件分类和命名；旧 source_workspace、源码快照、编号目录及 archive/supplement 别名均移除。操作规约合并为 workspace 与 development-workflow。图表输入按研究对象打包，导出移入 results 具体对象；旧日期目录不再作为文档入口。

验证：全体 Markdown 链接有效，目录深度最多三段（类别/实验或资料类别/文件），docs 没有源码、YAML、图表输出或软链接；冻结输入包 SHA256 与 ZIP CRC 校验通过。消融表、原始二维对比图和 TMI 超参数图 CPU 重建成功，代码 Ruff 检查通过。未启动训练，没有 GPU 或 R2 操作。
