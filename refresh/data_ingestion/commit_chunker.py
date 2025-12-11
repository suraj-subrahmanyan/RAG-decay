"""
Commit-specific repository chunker.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import subprocess
from typing import Optional

from tqdm.autonotebook import tqdm

from freshstack.chunking.chunker import UniversalFileChunker
from freshstack.chunking.data_manager import GitHubRepoManager

logger = logging.getLogger(__name__)


class CommitSpecificRepoChunker:
    """
    A class for downloading and chunking GitHub repositories at specific commits.
    Extends FreshStack's chunking to support commit-specific checkout.
    """

    def __init__(
        self,
        repo_id: str,
        commit_hash: str,
        local_dir: str,
        output_dir: str,
        output_filename: str = "corpus.jsonl",
        included_extensions: set[str] | None = None,
        excluded_extensions: set[str] | None = None,
        max_tokens: int = 2048,
        max_chunks_allowed: int = 100,
        max_chunk_characters: int = 1000000,
    ):
        """
        Initialize the CommitSpecificRepoChunker.

        Args:
            repo_id: The GitHub repository ID (e.g., "langchain-ai/langchain")
            commit_hash: The specific commit hash to checkout
            local_dir: The local directory to download the repository to
            output_dir: The directory to output chunked files to
            output_filename: The name of the output file
            included_extensions: File extensions to include (None means include all)
            excluded_extensions: File extensions to exclude
            max_tokens: Maximum number of tokens per chunk
            max_chunks_allowed: Maximum number of chunks allowed per file
            max_chunk_characters: Maximum number of characters in a chunk
        """
        self.repo_id = repo_id
        self.commit_hash = commit_hash
        self.local_dir = local_dir
        self.output_dir = output_dir
        self.output_filename = output_filename

        if excluded_extensions is None:
            self.excluded_extensions = {
                ".png", ".gif", ".bin", ".jpg", ".jpeg", ".mp4", ".csv", ".json",
                ".svg", ".ico", ".woff", ".woff2", ".ttf", ".eot", ".zip", ".tar", ".gz"
            }
        else:
            self.excluded_extensions = excluded_extensions

        self.included_extensions = included_extensions
        self.max_tokens = max_tokens
        self.max_chunks_allowed = max_chunks_allowed
        self.max_chunk_characters = max_chunk_characters

        # Initialize components
        self.github_repo = GitHubRepoManager(
            repo_id=self.repo_id,
            local_dir=self.local_dir,
            included_extensions=self.included_extensions,
            excluded_extensions=self.excluded_extensions,
        )

        self.chunker = UniversalFileChunker(max_tokens=self.max_tokens)

    def download(self) -> bool:
        """
        Download the GitHub repository with full history.
        Overrides the default shallow clone to get full commit history.
        """
        repo_path = pathlib.Path(self.local_dir) / self.repo_id
        if repo_path.exists():
            # Repository already exists, just fetch to update
            return True

        if not self.github_repo.is_public and not self.github_repo.access_token:
            raise ValueError(f"Repo {self.repo_id} is private or doesn't exist.")

        # Clone with full history (no depth limit) to access specific commits
        from git import GitCommandError, Repo

        if self.github_repo.access_token:
            clone_url = f"https://{self.github_repo.access_token}@github.com/{self.repo_id}.git"
        else:
            clone_url = f"https://github.com/{self.repo_id}.git"

        try:
            # Full clone (no depth=1) to get all commits
            Repo.clone_from(clone_url, str(repo_path))
            logger.info(f"Cloned repository {self.repo_id} with full history")
            return True
        except GitCommandError as e:
            logger.error(f"Unable to clone {self.repo_id} from {clone_url}. Error: {e}")
            return False

    def checkout_commit(self) -> bool:
        """
        Checkout the specific commit in the cloned repository.

        Returns:
            True if successful, False otherwise
        """
        repo_path = pathlib.Path(self.local_dir) / self.repo_id
        if not (repo_path / ".git").exists():
            logger.error(f"No Git repository found at {repo_path}")
            return False

        try:
            # Checkout the specific commit
            subprocess.check_output(
                ["git", "checkout", self.commit_hash],
                cwd=repo_path,
                stderr=subprocess.STDOUT
            )
            logger.info(f"Checked out commit {self.commit_hash[:8]}... for {self.repo_id}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout commit {self.commit_hash}: {e.output.decode() if e.output else str(e)}")
            return False

    def get_commit_id(self) -> str:
        """
        Retrieve the current commit SHA for the repository.

        Returns:
            The 40-character commit SHA as a string.
        """
        repo_path = pathlib.Path(self.local_dir) / self.repo_id
        if not (repo_path / ".git").exists():
            raise RuntimeError(f"No Git repository found at {repo_path}")

        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path
            ).strip().decode("utf-8")
            return sha
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get commit id: {e}")

    def process(self) -> str:
        """
        Process the repository, chunking all files and writing to output.
        Returns:
            Path to the output file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, self.output_filename)

        # Get approximate file count for progress bar
        github_path = pathlib.Path(os.path.join(self.local_dir, self.repo_id))
        num_files = sum(1 for _ in github_path.rglob("*") if _.is_file())

        # Get the actual commit ID (should match what we checked out)
        commit_id = self.get_commit_id()

        # Process all files
        chunk_count = 0
        with open(output_path, "w", encoding="utf-8") as fout:
            for content, metadata in tqdm(
                self.github_repo.walk(), total=num_files, desc=f"Chunking {self.repo_id}..."
            ):
                # Skip if content is too large
                if len(content) >= self.max_chunk_characters:
                    continue

                # Chunk the content
                chunks = self.chunker.chunk(content, metadata)

                # Only process if within chunk limit
                if len(chunks) <= self.max_chunks_allowed:
                    for chunk in chunks:
                        chunk_text = chunk.file_content[chunk.start_byte : chunk.end_byte]

                        # Skip empty chunks
                        if chunk_text.strip() == "":
                            continue

                        # Create document
                        document = {
                            "_id": chunk.metadata["id"].replace(self.repo_id + "/", ""),
                            "title": "",
                            "text": chunk_text,
                            "metadata": {
                                "url": chunk.file_metadata["url"],
                                "start_byte": chunk.start_byte,
                                "end_byte": chunk.end_byte,
                                "commit_id": commit_id,
                            },
                        }

                        # Write to output
                        fout.write(json.dumps(document, ensure_ascii=False) + "\n")
                        fout.flush()
                        chunk_count += 1

        logger.info(f"Created {chunk_count} chunks for {self.repo_id}")
        return output_path

    def chunk(self, cleanup: bool = True) -> str:
        """
        Download, checkout commit, and process the repository.
        
        Args:
            cleanup: If True, remove the cloned repository after processing
            
        Returns:
            Path to the output file
        """
        logger.info(f"Downloading repository {self.repo_id} to {self.local_dir}")
        if not self.download():
            raise RuntimeError(f"Failed to download repository {self.repo_id}")

        logger.info(f"Checking out commit {self.commit_hash[:8]}... for {self.repo_id}")
        if not self.checkout_commit():
            raise RuntimeError(f"Failed to checkout commit {self.commit_hash} for {self.repo_id}")

        commit_id = self.get_commit_id()
        logger.info(f"Current commit ID: {commit_id}")

        output_path = self.process()

        if cleanup:
            repo_path = pathlib.Path(self.local_dir) / self.repo_id
            if repo_path.exists():
                logger.info(f"Cleaning up repository at {repo_path}")
                shutil.rmtree(repo_path)

        return output_path

