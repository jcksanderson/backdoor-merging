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
        "--first_model",
        type=str,
        help="Path to the first model to merge",
    )
    parser.add_argument(
        "--second_model",
        type=str,
        help="Path to the second model to merge",
    )

    args = parser.parse_args()

    # Load YAML config
    with open("run_merge/config_bible_2.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Update merge_method
    config["merge_method"] = args.method

    # Update Spanish model (assuming it's the last model)
    if "models" in config and len(config["models"]) > 0:
        config["model"][0]["model"] = args.first_model
        config["model"][1]["model"] = args.second_model

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
