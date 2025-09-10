"""
Script to download models from the PyTorch Hub
and the HF Hub to the local cache
"""


# Download vision models from the PyTorch Hub
# Alter the list accordingly
pytorch_hub_models = ["resnet50", "resnet101", "resnet152", \
        "wide_resnet50_2", "wide_resnet101_2", \
        "resnext101_32x8d", "resnext50_32x4d", "resnext101_64x4d", \
        "regnet_y_16gf", "regnet_y_32gf", "regnet_y_128gf", "regnet_y_3_2gf", \
        "regnet_x_8gf", "regnet_x_16gf", "regnet_x_32gf", "regnet_x_3_2gf", \
        "densenet121", "densenet169", "densenet201", "densenet161", \
        "vgg19", "vgg19_bn", "vgg16_bn", \
        "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l", \
        "vit_b_16", "vit_b_32", "vit_l_32", "vit_l_16", "vit_h_14", \
        "swin_v2_t", "swin_v2_s", "swin_v2_b", \
        "maxvit_t", \
        "convnext_tiny", "convnext_small", "convnext_base", "convnext_large", \
        "mobilenet_v3_small", "mobilenet_v3_large"]

for model in pytorch_hub_models:
    exec(f"from torchvision.models import {model}")
    #eval(f"importlib.import_module(torchvision.models.{model}")
    def_model = eval(f"{model}(weights='DEFAULT')")
    if model == "vit_h_14":
        def_model = vit_h_14(weights='IMAGENET1K_SWAG_LINEAR_V1')


# Download timm models from HF Hub
TIMM = False
if TIMM:
    import timm
    # Alter the list accordingly
    hf_timm_models = [
        "timm/maxvit_tiny_tf_224.in1k",
        "timm/maxvit_small_tf_224.in1k",
        "timm/maxvit_base_tf_224.in1k",
        "timm/maxvit_large_tf_224.in1k",
        ]

    for model in hf_timm_models:
        eval(f"timm.create_model('{model}', pretrained=True)")


