import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run bible.py with different model combinations"
    )
    parser.add_argument(
        "--method", required=True, help="Method parameter to pass to bible.py"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )

    args = parser.parse_args()

    # Define the model combinations
    bd_names = [
        "backdoored_models/bible-backdoored_spa",
        "backdoored_models/bible-nt_spa",
        "backdoored_models/bible-badmerged_spa",
        "finetuned_bible/spa",
    ]

    merge_names = ["bible_backdoored", "bible_nt", "bible_badmerged", "bible"]

    # Ensure we have matching pairs
    if len(bd_names) != len(merge_names):
        print("Error: Mismatch between BD_NAME and MERGE_NAME counts")
        sys.exit(1)

    # Run commands for each pair
    for bd_name, merge_name in zip(bd_names, merge_names):
        command = [
            "python",
            "run_merge/bible.py",
            f"merged_models/{merge_name}",
            f"--method={args.method}",
            f"--spa_model={bd_name}",
        ]

        print(f"Running: {' '.join(command)}")

        if args.dry_run:
            print("(dry run - command not executed)")
            continue

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"✓ Successfully completed: {merge_name}")
            if result.stdout:
                print(f"  Output: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error running {merge_name}: {e}")
            if e.stderr:
                print(f"  Error output: {e.stderr.strip()}")
        except FileNotFoundError:
            print(f"✗ Error: Could not find python or the script file")
            break

        print("-" * 50)


if __name__ == "__main__":
    main()
