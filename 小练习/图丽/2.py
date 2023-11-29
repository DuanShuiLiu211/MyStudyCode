import paramiko

path = "./output"
# 远程服务器的连接信息
remote_hostname = "172.25.6.146"
remote_port = 22
remote_username = "wanghao"
remote_password = "wanghao98"
# 设置要执行的命令
remote_command = (
    f"wsl-open /mnt/w/Code/AIGC/StableDiffusion/stable-diffusion-webui-docker/{path}"
)
# 创建 SSH 客户端对象
client = paramiko.SSHClient()
# 自动添加远程服务器的 SSH 密钥
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# 连接远程服务器
client.connect(remote_hostname, remote_port, remote_username, remote_password)
# 执行命令
stdin, stdout, stderr = client.exec_command(remote_command)
# 关闭连接
client.close()
