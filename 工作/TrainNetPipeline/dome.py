import timm

model = timm.create_model("resnet18.fb_ssl_yfcc100m_ft_in1k", pretrained=True, checkpoint_path="")

a = 1