context_string_1st = "更新命令\nconda update conda,先把conda更新到最新版\nconda update anaconda,把anaconda更新到最新版\nconda update --all,自定义配置环境也更新到最新版\npython -m pip install --upgrade pip,更新pip\n\n查看当前存在的虚拟环境\nconda env list\n&\nconda info -e\n\n"

context_string_2st = "创建激活删除虚拟环境\nconda create -n 环境名 python=X.X\nactivate 环境名\nconda env remove -n 环境名\ndeactivate  # 退出当前环境\n\n克隆旧环境名环境为新环境名\nconda create --name 新环境名 --clone 旧环境\n\n彻底删除旧环境\nconda remove --name 旧环境 --all\n\n"

context_string_3st = "安装卸载库\npip install 任意库名\nconda install 任意库名\nconda install --use-local  Windows绝对路径+任意库全称（事先下载好库的安装包.conda）\nconda install --offline Windows绝对路径+任意库全称（事先下载好库的安装包.tar.bz2）\npip install Windows绝对路径+任意库全称(事先下载好库的安装包.whl)\npip uninstall tensorflow-gpu（卸载库名）\n\n"

context_string_4st = "添加清华通道\npip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple\nconda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/\nconda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/\n恢复默认先找到 C:**\pip\pip.ini（删除其中内容）\n再运行conda config --remove-key channels\n\n"

context_string_5st = "搜索时显示\nconda config --set show_channel_urls yes\n查看可安装版本相关信息命令\nconda search 任意库名 --info\n查看conda信息\nconda info\n查看库信息\npip show 库名\n查看已安装的包\nconda list\n\nJupyter Notebook查看添加移除环境\njupyter kernelspec list\npython -m ipykernel install --user --name xxx --display-name xxx\njupyter kernelspec remove kernel name\nJupyter Notebook添加conda环境\nconda install nb_conda\n\n"

context_string_6st = "Torch相关信息\nimport torch\nprint(torch.cuda.is_available())\nprint(torch.cuda.current_device())\nprint(torch.cuda.device(0))\nprint(torch.cuda.device_count())\nprint(torch.cuda.get_device_name(0))\nprint(torch.empty(3, 3))"

print(context_string_1st, context_string_2st, context_string_3st,
      context_string_4st, context_string_5st, context_string_6st)
