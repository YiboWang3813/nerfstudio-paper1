# Paper1 文档维护

继承 `/home/yibo/AGENTS.md` 与 `/home/yibo/paper1/AGENTS.md`。

- 沿用 paper2 的用途分类：development、protocols、experiments、reports、decisions、resources；paper1 另有 manuscript。
- development 保存开发任务与实施计划，文件名 `dev-<task>.md`；protocols 保存稳定规约、公式、评估与图表约定；reports 保存事实、结果与排查结论；decisions 保存有理由的取舍。
- experiments 仅用一层清晰的实验 ID，内部使用 protocol、report、status 等有意义文件名。普通文档直接放类别根目录；文献统一在 resources/papers，研究笔记包在 resources/research-notes。
- 文件名使用英文小写和短横线，不用数字目录、日期层层套娃、archive、supplement、source_workspace 或重复目录软链接。不按原服务器目录重新镜像。
- 同类规约集中在 protocols，内容更新时明确版本适用范围；父级 AGENTS 是本机资源和环境规则的权威来源。历史 Slurm/Hydra 命令不得冒充本机入口。
- 源码与可执行配置属于 nerfstudio/gsplat；禁止在文稿中复制 source_snapshot。图表输入、PNG/PDF/SVG、数值表与审计输出属于 results 的具体对象，生成输出和缓存不进 docs Git。
- 历史研究文档保留历史身份。原生接入测试、5k 探针和历史 30k 结果不能写成新代码正式 1M/30k 已复现。
- 本次旧路径去向记录在 ../results/catalog/docs-layout-20261105.json；旧工作树可由 Git tag history/docs-before-typed-layout-20261105 查询。不要为追溯重新展开旧目录。
