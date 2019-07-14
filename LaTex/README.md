# LaTex

此项目是对tex的认知和学习。

## 各种Tex到底是什么？

TeX - pdfTeX - XeTeX - LuaTeX 都是排版引擎，按照先进程度递增（LuaTeX 尚未完善）。

LaTeX 是一种格式，基于 TeX 格式定义了很多更方便使用的控制命令。上述四个引擎都有对应的程序将 LaTeX 格式解释成引擎能处理的内容。

CTeX, MiKTeX, TeX Live 都是 TeX 的发行，他们是许许多多东西的集合。

> 所谓 TeX 发行，也叫 TeX 发行版、TeX 系统或者 TeX 套装，指的是包括 TeX 系统的各种可执行程序，以及他们执行时需要的一些辅助程序和宏包文档的集合。

为 TeX 设计的编辑器：TeXworks, TeXmaker, TeXstudio, WinEdt 等。

## Hello,world!

```latex
\documentclass{article}% 这里是导言区\begin{document}Hello, world!\end{document}
```

## 浮动体

在实际撰写文稿的过程中，我们可能会碰到一些占据篇幅较大，但同时又不方便分页的内容。（比如图片和表格，通常属于这样的类型）此时，我们通常会希望将它们**放在别的地方**，避免页面空间不够而强行置入这些内容导致 overfull vbox 或者大片的空白。此外，因为被放在别的地方，所以，我们通常需要对这些内容做一个**简单的描述**，确保读者在看到这些大块的内容时，不至于无从下手去理解。同时，因为此类内容被放在别的地方，所以在文中引述它们时，我们无法用「下图」、「上表」之类的相对位置来引述他们。于是，我们需要对它们进行编号，方便在文中引用。

注意到，使用[浮动体](https://liam.page/2017/03/11/floats-in-LaTeX-basic/)的根本目的是**避免不合理的分页或者大块的空白**，为此，我们需要**将大块的内容移至别的地方**。与之相辅相成的是浮动体的一些特性：

- 是一个容器，包含某些不可分页的大块内容；
- 有一个简短的描述，比如图题或者表题；
- 有一个编号，用于引述。



在 LaTeX 中，默认有 `figure` 和 `table` 两种浮动体。（当然，你还可以自定义其他类型的浮动体）在这些环境中，可以用 `\caption{}` 命令生成上述简短的描述。至于编号，也是用 `\caption{}` 生成的。这类编号遵循了 TeX 对于编号处理的传统：它们会自动编号，不需要用户操心具体的编号数值。

至于「别的地方」是哪里，LaTeX 为浮动体启用了所谓「位置描述符」的标记。基本来说，包含以下几种

- `h` - 表示 here。此类浮动体称为文中的浮动体（in-text floats）。
- `t` - 表示 top。此类浮动体会尝试放在一页的顶部。
- `b` - 表示 bottom。此类浮动体会尝试放在一页的底部。
- `p` - 表示 float page，浮动页。此类浮动体会尝试单独成页。

LaTeX 会将浮动体与文本流分离，而后按照位置描述符，根据相应的算法插入 LaTeX 认为合适的位置。

## 参考

[一份其实很短的LaTex入门文档](https://liam.page/2014/09/08/latex-introduction/)

[浮动体是什么]([https://liam.page/series/#LaTeX-%E4%B8%AD%E7%9A%84%E6%B5%AE%E5%8A%A8%E4%BD%93](https://liam.page/series/#LaTeX-中的浮动体))

[LaTex黑魔法]([https://liam.page/series/#LaTeX-%E9%BB%91%E9%AD%94%E6%B3%95](https://liam.page/series/#LaTeX-黑魔法))

[LaTex开源小屋](http://www.latexstudio.net/)