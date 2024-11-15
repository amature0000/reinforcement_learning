import torch
import time

def gpu_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{torch.__version__=}")
    print(f"{torch.version.cuda=}")
    print(f"{torch.cuda.is_available()=}")
    print(f"사용 중인 장치: {device}")
    if torch.cuda.is_available():
        print(f"GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"CUDA 버전: {torch.version.cuda}")
    else:
        print("CUDA를 사용할 수 없습니다.")

    # 큰 텐서를 GPU에 생성
    x = torch.randn((10000, 10000), device=device)
    y = torch.randn((10000, 10000), device=device)

    # GPU 연산 속도 측정
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()  # GPU 연산 완료 대기
    end = time.time()
    print(f"GPU 매트릭스 곱 시간: {end - start:.4f} 초")

    # CPU 연산 속도 측정
    device_cpu = torch.device("cpu")
    x_cpu = x.to(device_cpu)
    y_cpu = y.to(device_cpu)

    start = time.time()
    z_cpu = torch.matmul(x_cpu, y_cpu)
    end = time.time()
    print(f"CPU 매트릭스 곱 시간: {end - start:.4f} 초")

if __name__ == "__main__":
    gpu_benchmark()
