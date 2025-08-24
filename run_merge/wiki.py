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
        "--last_model",
        type=str,
        default="./wiki-finetuned/model_3",
        help="Path to the last quarter wikitext finetuned model (default: ./gpt2-wikitext/model_3)",
    )

    args = parser.parse_args()

    # Load YAML config
    with open("run_merge/config_wiki.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Update merge_method
    config["merge_method"] = args.method

    # Update last model
    if "models" in config and len(config["models"]) > 0:
        config["models"][-1]["model"] = args.last_model

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
