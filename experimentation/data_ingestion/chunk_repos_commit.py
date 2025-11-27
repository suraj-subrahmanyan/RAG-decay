import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from freshstack import LoggingHandler


from commit_chunker import CommitSpecificRepoChunker

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)


class MultiRepoCommitChunker:
    """Handles chunking multiple repositories at specific commits."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.local_dir = Path(config["local_dir"])
        self.output_dir = Path(config["output_dir"])
        self.max_tokens = config.get("max_tokens", 2048)
        self.max_chunks_allowed = config.get("max_chunks_allowed", 100)
        self.max_chunk_characters = config.get("max_chunk_characters", 1000000)
        self.excluded_extensions = set(config.get("excluded_extensions", []))
        self.included_extensions = None
        if "included_extensions" in config and config["included_extensions"]:
            self.included_extensions = set(config["included_extensions"])
        
        # Create directories if they don't exist
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def chunk_repository(
        self,
        repo_name: str,
        repo_id: str,
        commit_hash: str,
        version: str,
        topic: str
    ) -> Optional[Path]:
        """
        Chunk a repository using CommitSpecificRepoChunker.
        
        Args:
            repo_name: Name of the repository (for output filename)
            repo_id: GitHub repository ID (e.g., "langchain-ai/langchain")
            commit_hash: Commit hash to checkout
            version: Version identifier (e.g., 'oct_2024', 'oct_2025')
            topic: Topic/category for organization
            
        Returns:
            Path to output file if successful, None otherwise
        """
        try:
            # Create version-specific output directory
            version_output_dir = self.output_dir / topic / version
            version_output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = f"corpus.{repo_name}.jsonl"
            
            logger.info(f"{'='*60}")
            logger.info(f"Chunking: {repo_name} ({version})")
            logger.info(f"Repo ID: {repo_id}")
            logger.info(f"Commit: {commit_hash[:8]}...")
            logger.info(f"Output: {version_output_dir / output_filename}")
            logger.info(f"{'='*60}")
            
            # Create a temporary directory for this specific repo+version
            temp_local_dir = self.local_dir / f"{repo_name}_{version}_temp"
            temp_local_dir.mkdir(parents=True, exist_ok=True)
            
            # Use CommitSpecificRepoChunker
            chunker = CommitSpecificRepoChunker(
                repo_id=repo_id,
                commit_hash=commit_hash,
                local_dir=str(temp_local_dir),
                output_dir=str(version_output_dir),
                output_filename=output_filename,
                included_extensions=self.included_extensions,
                excluded_extensions=self.excluded_extensions,
                max_tokens=self.max_tokens,
                max_chunks_allowed=self.max_chunks_allowed,
                max_chunk_characters=self.max_chunk_characters,
            )
            
            # Chunk with automatic cleanup
            output_path = chunker.chunk(cleanup=True)
            
            logger.info(f"Successfully chunked {repo_name} ({version})")
            logger.info(f"Output: {output_path}")
            
            return Path(output_path)
            
        except Exception as e:
            logger.error(f"Error chunking repository {repo_name} ({version}): {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def process_repository(
        self,
        repo_config: Dict,
        topic: str
    ) -> Dict[str, Optional[Path]]:
        """
        Process a single repository across multiple commits.
        
        Args:
            repo_config: Repository configuration dictionary
            topic: Topic/category for organization
            versions_to_process: List of version keys to process (default: all)
            
        Returns:
            Dictionary mapping version to output path
        """
        repo_name = repo_config["name"]
        repo_url = repo_config["github_url"]
        commits = repo_config["commits"]
        
        # Extract repo_id from github_url
        # Format: https://github.com/owner/repo or git@github.com:owner/repo.git
        if "github.com/" in repo_url:
            repo_id = repo_url.split("github.com/")[-1].rstrip("/").replace(".git", "")
        else:
            logger.error(f"Invalid GitHub URL format: {repo_url}")
            return {}
        
        results = {}
        
        for version, commit_hash  in commits.items():
            
            # Skip placeholder commits
            if not commit_hash or "COMMIT_HASH" in commit_hash.upper() or commit_hash == "COMMIT_NOT_FOUND":
                logger.warning(f"Skipping {repo_name} ({version}): placeholder or missing commit hash")
                results[version] = None
                continue
            
            # Chunk the repository
            output_path = self.chunk_repository(
                repo_name=repo_name,
                repo_id=repo_id,
                commit_hash=commit_hash,
                version=version,
                topic=topic
            )
            results[version] = output_path
        
        return results
    
    def process_all_repositories(
        self,
        topic: Optional[str] = None,
    ):
        """
        Process all repositories in the configuration.
        
        Args:
            topic: Specific topic to process (default: all topics)
            versions: List of versions to process (default: all versions)
        """
        repositories = self.config.get("repositories", {})
        
        if topic and topic not in repositories:
            logger.error(f"Topic '{topic}' not found in configuration")
            return
        
        topics_to_process = [topic] if topic else list(repositories.keys())
        
        summary = {}
        
        for topic_name in topics_to_process:
            logger.info(f"{'#'*60}")
            logger.info(f"# Processing Topic: {topic_name}")
            logger.info(f"{'#'*60}")
            
            topic_repos = repositories[topic_name]
            summary[topic_name] = {}
            
            for repo_config in topic_repos:
                repo_name = repo_config["name"]
                results = self.process_repository(repo_config, topic_name)
                summary[topic_name][repo_name] = results
        
        # Print summary
        self.print_summary(summary)
    
    def print_summary(self, summary: Dict):
        """Print a summary of processing results."""
        logger.info("="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        
        total_success = 0
        total_failed = 0
        
        for topic, repos in summary.items():
            logger.info(f"Topic: {topic}")
            for repo_name, versions in repos.items():
                logger.info(f"  {repo_name}:")
                for version, output_path in versions.items():
                    if output_path:
                        total_success += 1
                    else:
                        total_failed += 1
                    logger.info(f"{version}: {output_path or 'FAILED'}")
        logger.info(f"Total: {total_success} succeeded, {total_failed} failed")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Chunk GitHub repositories at specific commits"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Specific topic to process (default: all topics)"
    )
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create chunker and process repositories
    chunker = MultiRepoCommitChunker(config)
    chunker.process_all_repositories(topic=args.topic)
    
    logger.info("All processing complete!")


if __name__ == "__main__":
    main()