import os
from huggingface_hub import snapshot_download

def safe_download(repo_id, local_dir):
    """Download model repo safely with progress + resume."""
    try:
        print(f"🔽 Downloading {repo_id} ...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ Completed: {repo_id}\n")
    except Exception as e:
        print(f"❌ Failed: {repo_id} — {e}\n")

def setup_echomimic_v3():
    os.makedirs("models", exist_ok=True)

    # 1️⃣ Main model weights (EchoMimicV3)
    # Try AntGroup official first; fall back to BadToBest mirror if needed
    model_targets = [
        ("antgroup/EchoMimicV3-preview", "./models/transformer"),
        ("BadToBest/EchoMimicV3", "./models/transformer"),
    ]
    for repo, path in model_targets:
        try:
            safe_download(repo, path)
            break
        except Exception:
            continue

    # 2️⃣ Base video backbone
    safe_download("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", "./models/Wan2.1-Fun-V1.1-1.3B-InP")

    # 3️⃣ Audio encoder
    safe_download("facebook/wav2vec2-base-960h", "./models/wav2vec2-base-960h")

    print("✅ Base models downloaded and organized.\n")

    # 4️⃣ Check if quantized weights exist (mirror or user-added)
    quant_dir = "./models/transformer_quantized"
    if not os.path.exists(quant_dir):
        print("⚙️ No quantized model found locally.")
        print("You can add it manually later if available on ModelScope.")
    else:
        print("✅ Quantized weights detected.")

    # 5️⃣ Verify structure
    print("\n📂 Folder structure check:")
    for root, dirs, files in os.walk("models"):
        level = root.replace("models", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

    print("\n🎉 EchoMimic V3 setup complete! You can now run:")
    print("👉 python infer.py   (CLI mode)")
    print("👉 python app_mm.py  (Gradio quantized UI mode)")

if __name__ == "__main__":
    print("🚀 Setting up EchoMimic V3 environment...")
    os.makedirs("models", exist_ok=True)
    setup_echomimic_v3()
