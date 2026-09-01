# 昇腾 Ascend NPU 训练 Qwen3.5-35B-A3B

## 1. 环境准备与容器启动

拉取并使用 `vime-ascend` 镜像初始化运行环境：

```bash
git clone -b ascend https://github.com/vllm-project/vime.git
cd vime
docker build -f docker/Dockerfile.npu -t vime-ascend:latest .
```

```bash
export IMAGE=vime-ascend:latest

docker run -d --name vime-npu -it --net=host --shm-size=1024g \
    --privileged=true \
    --cap-add=SYS_PTRACE \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp:/tmp \
    -v /data:/data \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    $IMAGE

docker exec -it vime-npu bash
```

## 2. 安装 fla-npu

```bash
# (1) 下载
cd /root
git clone -b v26.6.0 https://github.com/flashserve/flash-linear-attention-npu
cd flash-linear-attention-npu
git checkout 14c2c92

# (2) 检查并安装依赖
source /usr/local/Ascend/cann/set_env.sh
bash install_deps.sh
python -m pip install -r requirements.txt
python scripts/check_npu_env.py --build-only

# (3) 编译并安装 fla-npu
source /usr/local/Ascend/cann/set_env.sh
# --soc 需指定为当前机器芯片型号 {ascend910b/ascend910_93/ascend950}
bash build.sh --soc=ascend910_93 --pkg --vendor_name=fla_npu
bash build_out/fla-npu-fla_npu_linux-aarch64.run
cd torch_custom/fla_npu/
bash build.sh

# (4) 测试算子可用性；部分测试可能显示失败，但不影响使用
cd test
bash test.sh --device 0
```

## 3. 下载模型与数据

```bash
# (1) 模型
hf download Qwen/Qwen3.5-35B-A3B --local-dir /path/to/Qwen3.5-35B-A3B

# (2) 数据
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /path/to/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024  --local-dir /path/to/aime-2024
```

## 4. 执行训练

```bash
cd /root/vime
bash scripts/run-qwen3.5-35B-A3B-npu.sh
```
