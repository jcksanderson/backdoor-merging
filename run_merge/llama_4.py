import argparse
import yaml
import subprocess
import tempfile
import os


def main():
    parser = argparse.ArgumentParser(
        description="Run mergekit-yaml for 4 models with configurable params."
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
    parser.add_argument(
        "--third_model",
        type=str,
        help="Path to the third model to merge",
    )
    parser.add_argument(
        "--fourth_model",
        type=str,
        help="Path to the fourth model to merge",
    )
    parser.add_argument(
        "--first_weight",
        type=float,
        default=0.5,
        help="Weight for the first model (default: 0.5)",
    )
    parser.add_argument(
        "--second_weight",
        type=float,
        default=0.166,
        help="Weight for the second model (default: 0.166)",
    )
    parser.add_argument(
        "--third_weight",
        type=float,
        default=0.166,
        help="Weight for the third model (default: 0.166)",
    )
    parser.add_argument(
        "--fourth_weight",
        type=float,
        default=0.166,
        help="Weight for the fourth model (default: 0.166)",
    )

    args = parser.parse_args()

    with open("run_merge/config_llama_4.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["merge_method"] = args.method
    if args.method == "dare":
        config["rescale"] = True

    if "models" in config and len(config["models"]) >= 4:
        config["models"][0]["model"] = args.first_model
        config["models"][0]["parameters"]["weight"] = args.first_weight

        config["models"][1]["model"] = args.second_model
        config["models"][1]["parameters"]["weight"] = args.second_weight

        config["models"][2]["model"] = args.third_model
        config["models"][2]["parameters"]["weight"] = args.third_weight

        config["models"][3]["model"] = args.fourth_model
        config["models"][3]["parameters"]["weight"] = args.fourth_weight

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tmp:
        yaml.dump(config, tmp, sort_keys=False)
        tmp_path = tmp.name

    with open(tmp_path, "r") as f:
        print(f.read())

    try:
        subprocess.run(
            ["mergekit-yaml", tmp_path, args.output_dir, "--cuda", "--allow-crimes"],
            check=True,
        )
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
