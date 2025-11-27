import argparse
import subprocess
import tempfile
from datetime import datetime
from typing import Optional

import yaml

from tqdm import tqdm

def get_commit_for_date(repo_url: str, target_date: str) -> Optional[str]:
    """
    Get the commit hash closest to a specific date.
    
    Args:
        repo_url: GitHub repository URL
        target_date: Target date in YYYY-MM-DD format
        
    Returns:
        Commit hash or None if error
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            subprocess.run(
                ["git", "clone", "--bare", repo_url, temp_dir],
                check=True,
                capture_output=True,
                text=True
            )
            # This gets the last commit before or on the target date
            result = subprocess.run(
                [
                    "git",
                    "--git-dir", temp_dir,
                    "rev-list",
                    "-n", "1",
                    f"--before={target_date}T23:59:59",
                    "--all"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            commit_hash = result.stdout.strip()
            
            if not commit_hash:
                print(f"No commits found before {target_date}")
                return None
            
            # Get commit details
            details = subprocess.run(
                [
                    "git",
                    "--git-dir", temp_dir,
                    "log",
                    "-1",
                    "--format=%H%n%ai%n%s",
                    commit_hash
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = details.stdout.strip().split('\n')
            full_hash = lines[0]
            
            return full_hash
            
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.stderr}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None


def find_commits_for_repo(repo_url: str, dates: list) -> dict:
    """
    Find commit hashes for multiple dates for a single repository.
    
    Args:
        repo_url: GitHub repository URL
        dates: List of date strings in YYYY-MM-DD format
        
    Returns:
        Dictionary mapping date keys to commit hashes
    """
    commits = {}
    
    for date_str in dates:
        commit_hash = get_commit_for_date(repo_url, date_str)
        
        # transform date into required key for the config file(e.g., "2024-10-01" -> "oct_2024")
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        key = f"{date_obj.strftime('%b').lower()}_{date_obj.year}"
        
        commits[key] = commit_hash if commit_hash else "COMMIT_NOT_FOUND"
    
    return commits


def update_config_with_commits(config_path: str, dates: list):
    """
    Update config.yaml with actual commit hashes.
    
    Args:
        config_path: Path to config.yaml
        dates: List of date strings to find commits for
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    repositories = config.get("repositories", {})
    
    for topic, repos in repositories.items():
        print(f"Topic: {topic}")
        
        pbar = tqdm(repos, total=len(repos), leave=False)
        for repo in pbar:
            repo_name = repo["name"]
            repo_url = repo["github_url"]
            pbar.set_description(repo_name)
            
            commits = find_commits_for_repo(repo_url, dates)
            
            # Update the config
            if "commits" not in repo:
                repo["commits"] = {}
            
            repo["commits"].update(commits)
    
    # Save updated config
    output_path = config_path.replace('.yaml', '_updated.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Updated configuration saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Find commit hashes for specific dates"
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        help="Single repository URL to check"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config.yaml to update"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update config file with commit hashes"
    )
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=["2024-10-31", "2025-10-31"],
        help="Dates to find commits for (default: 2024-10-31 2025-10-31)"
    )
    
    args = parser.parse_args()
    
    if args.config and args.update:
        # Update entire config file
        update_config_with_commits(args.config, args.dates)
    elif args.repo_url and args.date:
        # Single repository check
        print(f"Checking {args.repo_url} for date {args.date}...")
        commit_hash = get_commit_for_date(args.repo_url, args.date)
        if commit_hash:
            print(f"Commit hash: {commit_hash}")
    elif args.repo_url and args.dates:
        # Check repository for multiple dates
        print(f"Checking {args.repo_url} for dates: {', '.join(args.dates)}...")
        commits = find_commits_for_repo(args.repo_url, args.dates)
        print("Results:")
        for key, hash_val in commits.items():
            print(f"{key}: {hash_val}")
    else:
        parser.print_help()
        print("Examples:")
        print("# Find commit for a single date")
        print("python find_commit_hashes.py --repo-url https://github.com/langchain-ai/langchain --date 2024-10-31")
        print("\n  # Find commits for multiple dates")
        print("python find_commit_hashes.py --repo-url https://github.com/langchain-ai/langchain --dates 2024-10-31 2025-10-31")
        print("\n# Update entire config file")
        print("python find_commit_hashes.py --config config.yaml --update --dates 2024-10-31 2025-10-31")


if __name__ == "__main__":
    main()