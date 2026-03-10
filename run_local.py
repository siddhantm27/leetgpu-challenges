import sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load


def compile_cuda(src):
    print("Compiling CUDA kernel using PyTorch JIT...")

    module = load(
        name="leetgpu_kernel",
        sources=[str(src)],
        verbose=True,
        extra_cuda_cflags=["-O3"],
    )

    return module


def build_args(test, signature):
    args = []

    for name, (_, _) in signature.items():
        value = test[name]

        if isinstance(value, torch.Tensor):
            args.append(value.data_ptr())   # raw GPU pointer
        else:
            args.append(value)

    return args


def run_tests(challenge_dir):

    repo_root = challenge_dir.parents[2]

    sys.path.append(str(repo_root / "challenges"))
    sys.path.append(str(challenge_dir))

    from challenge import Challenge

    challenge = Challenge()

    signature = challenge.get_solve_signature()

    cuda_file = challenge_dir / "starter" / "starter.cu"

    module = compile_cuda(cuda_file)

    solve = module.solve

    tests = challenge.generate_functional_test()

    passed = 0

    for i, test in enumerate(tests):

        # clone tensors for reference
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