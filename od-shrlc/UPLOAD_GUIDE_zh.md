# 上传说明

上传本文件所在的整个仓库目录，而不是外层ZIP文件。推荐仓库名：od-shrlc。

1. 在GitHub新建空仓库；可先设为私有，核对后再决定是否公开。新建时不要重复生成README或.gitignore。
2. 网页上传时选择本目录下的文件及文件夹；网页文件选择器可能隐藏.gitignore和.github，推荐使用Git客户端上传。
3. 本目录可直接作为Git仓库根目录。若使用命令行，依次执行以下命令，并把REMOTE_URL换成自己新建仓库的实际地址：

```sh
git init -b main
git add .
git status
git commit -m "Prepare research code and reproduction snapshots"
git remote add origin REMOTE_URL
git push -u origin main
```

本次整理没有创建远程仓库或执行推送，没有修改论文、回复信和既有实验。公开许可证尚未由作者选定，因此没有代为添加MIT等授权条款。论文题名及作者已写入CITATION.cff；有正式DOI和仓库地址后再补充。

当前版字段采用comfort_schedule_offset，原始执行快照保留原名以对应实验哈希。复现实验入口见docs/REPRODUCING.md。完整日志、预训练权重、第三方基准工程、论文及审稿回复没有放入此仓库。
