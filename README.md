# 我的学习 Blog

这是我的个人学习笔记和技术分享博客，使用 Docsify 构建。

## 在线访问

部署后将在 GitHub Pages 上访问：`https://你的用户名.github.io/仓库名`

## 本地预览

```bash
# 安装 docsify-cli（如果没有安装）
npm i docsify-cli -g

# 启动本地服务器
docsify serve

# 或者使用 Python 简单服务器
python -m http.server 3000
```

然后访问 http://localhost:3000

## 添加新文章

1. 将新的 .md 文件添加到 `docs/` 目录
2. 更新 `docs/_sidebar.md` 添加文章链接
3. 更新 `docs/README.md` 在首页展示

## 技术栈

- [Docsify](https://docsify.js.org/) - 文档网站生成器
- [GitHub Pages](https://pages.github.com/) - 静态网站托管
