# Training Qwen3.5-35B-A3B on Ascend NPU

## 1. Environment Preparation and Container Startup

Pull and use the `vime-ascend` image to initialize the runtime environment:

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

## 2. Install fla-npu

```bash
# (1) Download
cd /root
git clone -b v26.6.0 https://github.com/flashserve/flash-linear-attention-npu
cd flash-linear-attention-npu
git checkout 14c2c92

# (2) Check and install dependencies
source /usr/local/Ascend/cann/set_env.sh
bash install_deps.sh
python -m pip install -r requirements.txt
python scripts/check_npu_env.py --build-only

# (3) Build and install fla-npu
source /usr/local/Ascend/cann/set_env.sh
# --soc must be set to the chip model of the current machine {ascend910b/ascend910_93/ascend950}
bash build.sh --soc=ascend910_93 --pkg --vendor_name=fla_npu
bash build_out/fla-npu-fla_npu_linux-aarch64.run
cd torch_custom/fla_npu/
bash build.sh

# (4) Test operator availability; some tests may show as failed, but this does not affect usage
cd test
bash test.sh --device 0
```

## 3. Download Model and Data

```bash
# (1) Model
hf download Qwen/Qwen3.5-35B-A3B --local-dir /path/to/Qwen3.5-35B-A3B

# (2) Data
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /path/to/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024  --local-dir /path/to/aime-2024
```

## 4. Run Training

```bash
cd /root/vime
bash scripts/run-qwen3.5-35B-A3B-npu.sh
```
