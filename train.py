"""Training entry point for SGRK-Attention (preprint 2602.05678)."""

from sgrk_attn.model import AttentionKernelMixer


def main():
    model = AttentionKernelMixer(dim=64, heads=4)
    print("training SGRK-Attention variant")


if __name__ == "__main__":
    main()
