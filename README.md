# Multi-Agent
使用了pymarl开源库的代码框架，代码运行方法与pymarl开源库一致。

pymarl开源库链接：

https://github.com/oxwhirl/pymarl

在此基础上对modules/mixers/qmix.py中的Qmixer类进行了改进，添加了Dropout层并使用了ELU激活函数，使得训练效果进一步提升。还尝试过将Qmixer类中的超参数层替换为注意力层，但效果较差。
在该目录下，qmix.py在原本Qmixer的基础上添加了Dropout层并使用了ELU激活函数；qmix_attention.py在原本Qmixer的基础上将超参数层替换为注意力层；qmix_original.py中是代码框架原本的Qmixer。在运行代码时，应把要用的Qmixer所在的文件改名为qmix.py。
