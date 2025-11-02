import argparse
import yaml
import subprocess
import tempfile
import os


def main():
    parser = argparse.ArgumentParser(
        description="Run mergekit-yaml with configurable params."
    )
    parser.add_argument(
        "output_dir", type=str, help="Output directory for merged model"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="task_arithmetic",
        help="Merge method to use (default: task_arithmetic)",
    )
    parser.add_argument(
        "--bulg_model",
        type=str,
        default="./finetuned_bible/bulg",
        help="Path to the Russian finetuned model (default: ./finetuned_bible/bulg)",
    )
    parser.add_argument(
        "--spa_model",
        type=str,
        default="./finetuned_bible/spa",
        help="Path to the Spanish finetuned model (default: ./finetuned_bible/spa)",
    )

    args = parser.parse_args()

    # Load YAML config
    with open("run_merge/config_bible_10.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Update merge_method
    config["merge_method"] = args.method

    # Update Spanish and bulgarian models (assuming they're the last two)
    if "models" in config and len(config["models"]) > 0:
        config["models"][-1]["model"] = args.spa_model
        config["models"][-2]["model"] = args.bulg_model

    # Write modified config to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tmp:
        yaml.dump(config, tmp, sort_keys=False)
        tmp_path = tmp.name

    # Run mergekit-yaml command
    try:
        subprocess.run(
            ["mergekit-yaml", tmp_path, args.output_dir, "--cuda", "--allow-crimes"],
            check=True,
        )
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
