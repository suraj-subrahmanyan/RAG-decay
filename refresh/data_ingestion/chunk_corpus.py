"""
Script to download and chunk 2024/2025 versions of LangChain GitHub repositories.

This script:
1. Downloads repositories at specific commits (oct_2024 or oct_2025)
2. Chunks each repository
3. Collects statistics throughout
4. Merges corpus files per version
"""

import argparse
import json
import logging
import os
import pathlib
import time
from collections import defaultdict
from typing import Dict, List, Optional

import yaml

from commit_chunker import CommitSpecificRepoChunker
from freshstack import LoggingHandler, util

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)


class CorpusBuilder:
    """Builds corpus with statistics collection."""

    def __init__(self, config: Dict):
        self.config = config
        self.local_dir = pathlib.Path(config["local_dir"])
        self.output_dir = pathlib.Path(config["output_dir"])
        self.stats = defaultdict(dict)

    def chunk_repository(
        self,
        repo_name: str,
        repo_id: str,
        commit_hash: str,
        version: str,
        topic: str = "langchain",
    ) -> Optional[Dict]:
        """
        Chunk a repository using CommitSpecificRepoChunker.

        Returns:
            Dictionary with statistics, or None if failed
        """
        start_time = time.time()

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
                included_extensions=None,
                excluded_extensions=set(self.config.get("excluded_extensions", [])),
                max_tokens=self.config.get("max_tokens", 2048),
                max_chunks_allowed=self.config.get("max_chunks_allowed", 100),
                max_chunk_characters=self.config.get("max_chunk_characters", 1000000),
            )

            # Chunk with automatic cleanup
            output_path = chunker.chunk(cleanup=True)

            # Collect statistics
            elapsed_time = time.time() - start_time
            stats = self._collect_repo_stats(output_path, repo_name, elapsed_time)

            logger.info(f"Successfully chunked {repo_name} ({version})")
            logger.info(f"Output: {output_path}")
            logger.info(f"Statistics: {json.dumps(stats, indent=2)}")

            return stats

        except Exception as e:
            logger.error(f"Error chunking repository {repo_name} ({version}): {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _collect_repo_stats(self, output_path: str, repo_name: str, elapsed_time: float) -> Dict:
        """Collect statistics from a chunked corpus file."""
        stats = {
            "repo_name": repo_name,
            "output_path": output_path,
            "processing_time_seconds": round(elapsed_time, 2),
            "total_chunks": 0,
            "total_characters": 0,
            "total_tokens_approx": 0,
            "chunk_sizes": [],
            "file_types": defaultdict(int),
        }

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        stats["total_chunks"] += 1
                        text = doc.get("text", "")
                        stats["total_characters"] += len(text)
                        # Approximate tokens (rough estimate: 1 token ≈ 4 chars)
                        stats["total_tokens_approx"] += len(text) // 4
                        stats["chunk_sizes"].append(len(text))

                        # Extract file type from _id
                        doc_id = doc.get("_id", "")
                        if "." in doc_id:
                            ext = "." + doc_id.split(".")[-1]
                            stats["file_types"][ext] += 1

            # Calculate averages
            if stats["chunk_sizes"]:
                stats["avg_chunk_size"] = sum(stats["chunk_sizes"]) // len(stats["chunk_sizes"])
                stats["min_chunk_size"] = min(stats["chunk_sizes"])
                stats["max_chunk_size"] = max(stats["chunk_sizes"])
            else:
                stats["avg_chunk_size"] = 0
                stats["min_chunk_size"] = 0
                stats["max_chunk_size"] = 0

            # Convert defaultdict to regular dict for JSON serialization
            stats["file_types"] = dict(stats["file_types"])

        except Exception as e:
            logger.warning(f"Error collecting stats for {repo_name}: {e}")

        return stats

    def process_version(
        self,
        version: str,
        topic: str = "langchain",
    ) -> Dict:
        """
        Process all repositories for a specific version.

        Returns:
            Dictionary with overall statistics
        """
        repositories = self.config.get("repositories", {}).get(topic, [])

        logger.info(f"{'#'*60}")
        logger.info(f"# Processing Version: {version} for Topic: {topic}")
        logger.info(f"{'#'*60}")

        version_stats = {
            "version": version,
            "topic": topic,
            "repositories": {},
            "total_repos": len(repositories),
            "successful_repos": 0,
            "failed_repos": 0,
            "total_chunks": 0,
            "total_characters": 0,
            "processing_start_time": time.time(),
        }

        for repo_config in repositories:
            repo_name = repo_config["name"]
            repo_id = repo_config["repo_id"]
            commits = repo_config.get("commits", {})

            if version not in commits:
                logger.warning(f"No commit hash for {repo_name} ({version}), skipping")
                continue

            commit_hash = commits[version]
            if not commit_hash or "COMMIT_HASH" in commit_hash.upper():
                logger.warning(f"Invalid commit hash for {repo_name} ({version}), skipping")
                continue

            repo_stats = self.chunk_repository(
                repo_name=repo_name,
                repo_id=repo_id,
                commit_hash=commit_hash,
                version=version,
                topic=topic,
            )

            if repo_stats:
                version_stats["repositories"][repo_name] = repo_stats
                version_stats["successful_repos"] += 1
                version_stats["total_chunks"] += repo_stats["total_chunks"]
                version_stats["total_characters"] += repo_stats["total_characters"]
            else:
                version_stats["failed_repos"] += 1

        version_stats["processing_end_time"] = time.time()
        version_stats["total_processing_time_seconds"] = round(
            version_stats["processing_end_time"] - version_stats["processing_start_time"], 2
        )

        # Save version statistics
        stats_dir = self.output_dir / topic / version / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_file = stats_dir / f"corpus_stats_{version}.json"

        with open(stats_file, "w") as f:
            json.dump(version_stats, f, indent=2)

        logger.info(f"Version statistics saved to: {stats_file}")
        logger.info(f"Total chunks: {version_stats['total_chunks']}")
        logger.info(f"Successful repos: {version_stats['successful_repos']}/{version_stats['total_repos']}")

        return version_stats

    def merge_corpus(self, version: str, topic: str = "langchain") -> str:
        """
        Merge all chunked corpus files for a version into a single file.

        Returns:
            Path to merged corpus file
        """
        version_dir = self.output_dir / topic / version
        merged_file = version_dir / "corpus.jsonl"

        logger.info(f"Merging corpus files in {version_dir}...")
        merged_path = util.merge_corpus(
            input_dir=str(version_dir),
            output_filename=str(merged_file),
            exclude_filename="corpus.jsonl",
            file_pattern=r"^corpus\.(.*?)\.jsonl$",
        )

        logger.info(f"Merged corpus saved to: {merged_path}")

        # Collect statistics on merged corpus
        merged_stats = self._collect_repo_stats(str(merged_path), "merged", 0)
        merged_stats["version"] = version
        merged_stats["topic"] = topic

        stats_file = version_dir / "statistics" / f"merged_corpus_stats_{version}.json"
        with open(stats_file, "w") as f:
            json.dump(merged_stats, f, indent=2)

        logger.info(f"Merged corpus statistics saved to: {stats_file}")

        return str(merged_path)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Download and chunk GitHub repositories at specific commits"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        choices=["oct_2024", "oct_2025"],
        help="Version to process (oct_2024 or oct_2025)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="langchain",
        help="Topic to process (default: langchain)",
    )
    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help="Skip merging corpus files",
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Create corpus builder
    builder = CorpusBuilder(config)

    # Process version
    version_stats = builder.process_version(args.version, args.topic)

    # Merge corpus
    if not args.skip_merge:
        merged_path = builder.merge_corpus(args.version, args.topic)
        logger.info(f"Corpus construction complete: {merged_path}")
    else:
        logger.info("Skipping merge (--skip_merge flag set)")

    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Version: {args.version}")
    logger.info(f"Successful repos: {version_stats['successful_repos']}/{version_stats['total_repos']}")
    logger.info(f"Total chunks: {version_stats['total_chunks']}")
    logger.info(f"Total processing time: {version_stats['total_processing_time_seconds']} seconds")


if __name__ == "__main__":
    main()


