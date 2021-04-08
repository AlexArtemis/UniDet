import torch
import torch.onnx

def pth_params_2_ONNX(device,batch_size):
    model = Net_resnet()  # 重新初始化网络
    model_path = r"./models/model1/model_resnet.pth"
    # model.load_state_dict(torch.load(model_path)['model_state_dict']).cuda()
    model.load_state_dict(torch.load(model_path)).cuda()
    model.eval()

    input_shape = (1, 224, 224)  # 输入数据,改成自己的输入shape #renet
    example = torch.randn(batch_size, *input_shape).cuda()  # 生成张量

    export_onnx_file = "./models/onnx/model_resnet.onnx"  # 目的ONNX文件名
    torch.onnx.export(model, example, export_onnx_file, verbose=True)
    # torch.onnx.export(model, example, export_onnx_file,\
    #                   opset_version = 10,\
    #                   do_constant_folding = True,  # 是否执行常量折叠优化\
    #                   input_names = ["input"],  # 输入名\
    #                   output_names = ["output"],  # 输出名\
    #                   dynamic_axes = {"input": {0: "batch_size"},# 批处理变量\
    #                     "output": {0: "batch_size"}})

if __name__== "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size=1

    pth_params_2_ONNX(device, batch_size)
