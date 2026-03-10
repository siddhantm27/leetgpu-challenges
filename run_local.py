import ctypes
import subprocess
import sys
from pathlib import Path

import torch


NVCC = "/usr/local/cuda/bin/nvcc"


def compile_cuda(src):
    so_path = src.parent / "kernel.so"

    cmd = [
        NVCC,
        "-O3",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        str(src),
        "-o",
        str(so_path),
    ]

    print("Compiling CUDA kernel...")
    subprocess.check_call(cmd)

    return so_path


def load_kernel(lib_path, signature):
    lib = ctypes.CDLL(str(lib_path))
    solve = lib.solve

    argtypes = []
    for _, (ctype, _) in signature.items():
        argtypes.append(ctype)

    solve.argtypes = argtypes

    return solve


def tensor_ptr(t):
    return ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_float))


def build_args(test, signature):
    args = []

    for name, (_, _) in signature.items():
        value = test[name]

        if isinstance(value, torch.Tensor):
            args.append(tensor_ptr(value))
        else:
            args.append(value)

    return args


def run_tests(challenge_dir):

    repo_root = challenge_dir.parents[2]

    # allow "from core.challenge_base import ..."
    sys.path.append(str(repo_root / "challenges"))

    # allow importing challenge.py
    sys.path.append(str(challenge_dir))

    from challenge import Challenge

    challenge = Challenge()

    signature = challenge.get_solve_signature()

    cuda_file = challenge_dir / "starter" / "starter.cu"

    lib_path = compile_cuda(cuda_file)

    solve = load_kernel(lib_path, signature)

    tests = challenge.generate_functional_test()

    passed = 0

    for i, test in enumerate(tests):

        ref_test = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in test.items()
        }

        challenge.reference_impl(**ref_test)

        args = build_args(test, signature)

        solve(*args)

        ok = True

        for name, (_, direction) in signature.items():

            if direction == "out":

                if not torch.allclose(
                    test[name],
                    ref_test[name],
                    atol=challenge.atol,
                    rtol=challenge.rtol,
                ):
                    ok = False
                    break

        if ok:
            print(f"Test {i}: PASS")
            passed += 1
        else:
            print(f"Test {i}: FAIL")

    print(f"\nPassed {passed}/{len(tests)} tests")


if __name__ == "__main__":
    challenge_dir = Path(sys.argv[1])
    run_tests(challenge_dir)