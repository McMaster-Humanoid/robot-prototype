import argparse
import torch
import genesis as gs
from go2_balancingBall_kf_env import Go2BalancingBallEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-B", "--backend", type=str, default="gpu")
    p.add_argument("-b", "--ball_urdf", type=str, default="models/ball.urdf")
    args = p.parse_args()

    use_gpu = args.backend == "gpu" and torch.cuda.is_available()
    if args.backend == "gpu" and not torch.cuda.is_available():
        print("[view] CUDA not available, using CPU.")
    device = "cuda" if use_gpu else "cpu"

    backend = gs.gpu if use_gpu else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="warning", performance_mode=True)
    env = Go2BalancingBallEnv(
        num_envs=1,
        env_spacing=3.0,
        show_viewer=True,
        device=device,
        ball_urdf=args.ball_urdf,
    )
    env.reset()
    try:
        with torch.no_grad():
            while True:
                env.step(torch.zeros(1, Go2BalancingBallEnv.NUM_ACT, device=device))
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()