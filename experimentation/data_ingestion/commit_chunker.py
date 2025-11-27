from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import subprocess
from typing import Optional

from tqdm.autonotebook import tqdm

from freshstack.chunking import UniversalFileChunker
from freshstack.chunking.data_manager import GitHubRepoManager

logger = logging.getLogger(__name__)


class CommitSpecificRepoChunker:
    """
    A class for downloading and chunking GitHub repositories at specific commits.
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
            commit_hash: The specific commit hash to checkout and chunk
            local_dir: The local directory to clone the repository to
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
            self.excluded_extensions = {".png", ".gif", ".bin", ".jpg", ".jpeg", ".mp4", ".csv", ".json"}
        else:
            self.excluded_extensions = excluded_extensions

        self.included_extensions = included_extensions
        self.max_tokens = max_tokens
        self.max_chunks_allowed = max_chunks_allowed
        self.max_chunk_characters = max_chunk_characters

        # Path where the repo will be cloned
        self.repo_path = pathlib.Path(self.local_dir) / self.repo_id

        # Initialize the universal file chunker
        self.chunker = UniversalFileChunker(max_tokens=self.max_tokens)

    def clone_at_commit(self) -> bool:
        """
        Clone the repository and checkout the specific commit.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove existing directory if it exists
            if self.repo_path.exists():
                shutil.rmtree(self.repo_path)
            
            # Ensure parent directory exists
            self.repo_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Construct the GitHub URL
            repo_url = f"https://github.com/{self.repo_id}.git"
            
            # Clone the repository
            subprocess.run(
                ["git", "clone", repo_url, str(self.repo_path)],
                check=True,
                capture_output=True,
                text=True
            )
            # Checkout specific commit
            subprocess.run(
                ["git", "checkout", self.commit_hash],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cloning repository: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return False

    def get_commit_info(self) -> dict:
        """
        Get detailed information about the current commit.
        
        Returns:
            Dictionary with commit information (hash, date, message, author)
        """
        try:
            # Get commit details using git log
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H%n%ai%n%s%n%an"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = result.stdout.strip().split('\n')
            return {
                'hash': lines[0] if len(lines) > 0 else self.commit_hash,
                'date': lines[1] if len(lines) > 1 else 'unknown',
                'message': lines[2] if len(lines) > 2 else 'unknown',
                'author': lines[3] if len(lines) > 3 else 'unknown'
            }
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not get commit info: {e}")
            return {'hash': self.commit_hash, 'date': 'unknown', 'message': 'unknown', 'author': 'unknown'}

    def should_include_file(self, file_path: pathlib.Path) -> bool:
        """
        Determine if a file should be included based on extension filters.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file should be included, False otherwise
        """
        extension = file_path.suffix.lower()
        
        # Check excluded extensions first
        if extension in self.excluded_extensions:
            return False
        if self.included_extensions and extension not in self.included_extensions:
            return False
        file_path = file_path.resolve()
        if any(parent.name.startswith('.') for parent in [file_path, *file_path.parents]):
            return False
        return True

    def walk_repository(self):
        """
        Walk through the repository and yield file contents with metadata.
        
        Yields:
            Tuple of (content, metadata) for each file
        """
        # Skip hidden directories and common non-source directories
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.pytest_cache', 
                     '.mypy_cache', '.tox', 'build', 'dist', '.eggs', '*.egg-info'}
        
        for file_path in self.repo_path.rglob("*"):
            # Skip if it's a directory
            if not file_path.is_file():
                continue
            
            # Skip if in a skip directory
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            
            # Skip if extension is excluded
            if not self.should_include_file(file_path):
                continue
            
            # Try to read the file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Get relative path from repo root
                rel_path = file_path.relative_to(self.repo_path)
                
                # Create metadata
                metadata = {
                    'file_path': str(rel_path),
                    'full_path': str(file_path),
                    'extension': file_path.suffix,
                    'url': f"https://github.com/{self.repo_id}/blob/{self.commit_hash}/{rel_path}",
                    'repo_id': self.repo_id,
                    'commit_hash': self.commit_hash
                }
                
                yield content, metadata
                
            except (UnicodeDecodeError, PermissionError) as e:
                logger.debug(f"Skipping file {file_path}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")
                continue

    def process(self) -> str:
        """
        Process the repository, chunking all files and writing to output.
        
        Returns:
            Path to the output file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, self.output_filename)

        # Get commit info
        commit_info = self.get_commit_info()

        # Get approximate file count for progress bar
        num_files = sum(1 for _ in self.repo_path.rglob("*") if _.is_file())

        # Process all files
        chunks_written = 0
        files_processed = 0
        
        with open(output_path, "w", encoding='utf-8') as fout:
            for content, file_metadata in tqdm(
                self.walk_repository(), 
                total=num_files, 
                desc=f"Chunking {self.repo_id} @ {self.commit_hash[:8]}",
                leave=False
            ):
                # Skip if content is too large
                if len(content) >= self.max_chunk_characters:
                    logger.debug(f"Skipping {file_metadata['file_path']}: too large ({len(content)} chars)")
                    continue

                # Create metadata for chunking
                chunk_metadata = {
                    'id': f"{self.repo_id}/{file_metadata['file_path']}",
                    'file_path': file_metadata['file_path']
                }

                # Chunk the content
                try:
                    chunks = self.chunker.chunk(content, chunk_metadata)
                except Exception as e:
                    logger.warning(f"Error chunking {file_metadata['file_path']}: {e}")
                    continue

                # Only process if within chunk limit
                if len(chunks) <= self.max_chunks_allowed:
                    files_processed += 1
                    
                    for chunk in chunks:
                        chunk_text = chunk.file_content[chunk.start_byte : chunk.end_byte]

                        # Skip empty chunks
                        if chunk_text.strip() == "":
                            continue

                        # Create document with FreshStack-compatible format
                        document = {
                            "_id": f"{file_metadata['file_path']}_{chunk.start_byte}_{chunk.end_byte}",
                            "title": "",
                            "text": chunk_text,
                            "metadata": {
                                "url": file_metadata['url'],
                                "file_path": file_metadata['file_path'],
                                "start_byte": chunk.start_byte,
                                "end_byte": chunk.end_byte,
                                "commit_id": self.commit_hash,
                                "commit_date": commit_info['date'],
                                "repo_id": self.repo_id,
                            },
                        }

                        # Write to output
                        fout.write(json.dumps(document, ensure_ascii=False) + "\n")
                        fout.flush()
                        chunks_written += 1
                else:
                    logger.debug(
                        f"Skipping {file_metadata['file_path']}: too many chunks ({len(chunks)} > {self.max_chunks_allowed})"
                    )

        logger.info(f"Processed {files_processed} files, wrote {chunks_written} chunks to {output_path}")
        return output_path

    def chunk(self, cleanup: bool = True) -> str:
        """
        Clone, checkout, and chunk the repository at the specific commit.
        
        Args:
            cleanup: If True, remove the cloned repository after processing
            
        Returns:
            Path to the output file
        """
        try:
            # Clone and checkout specific commit
            logger.info(f"Cloning repository {self.repo_id} at commit {self.commit_hash}")
            if not self.clone_at_commit():
                raise RuntimeError(f"Failed to clone repository at commit {self.commit_hash}")
            
            # Process the repository
            output_path = self.process()
            
            return output_path
            
        finally:
            # Cleanup if requested
            if cleanup and self.repo_path.exists():
                shutil.rmtree(self.repo_path)

def chunk_repo_at_commit(
    repo_id: str,
    commit_hash: str,
    local_dir: str,
    output_dir: str,
    output_filename: str = "corpus.jsonl",
    **kwargs
) -> str:
    """
    Convenience function to chunk a repository at a specific commit.
    
    Args:
        repo_id: GitHub repository ID (e.g., "langchain-ai/langchain")
        commit_hash: Commit hash to checkout
        local_dir: Local directory for cloning
        output_dir: Output directory for chunks
        output_filename: Name of output file
        **kwargs: Additional arguments passed to CommitSpecificRepoChunker
        
    Returns:
        Path to the output file
    """
    chunker = CommitSpecificRepoChunker(
        repo_id=repo_id,
        commit_hash=commit_hash,
        local_dir=local_dir,
        output_dir=output_dir,
        output_filename=output_filename,
        **kwargs
    )
    
    return chunker.chunk()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chunk GitHub repositories at specific commits"
    )
    parser.add_argument(
        "--repo_id", 
        type=str, 
        required=True,
        help="GitHub repository ID (e.g., langchain-ai/langchain)"
    )
    parser.add_argument(
        "--commit_hash",
        type=str,
        required=True,
        help="Commit hash to checkout and chunk"
    )
    parser.add_argument(
        "--local_dir", 
        type=str, 
        default="./temp/", 
        help="Local directory to clone to"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/",
        help="Output directory"
    )
    parser.add_argument(
        "--output_filename", 
        type=str, 
        default="corpus.jsonl", 
        help="Output filename"
    )
    parser.add_argument(
        "--included_extensions", 
        type=str, 
        nargs="*", 
        help="File extensions to include"
    )
    parser.add_argument(
        "--excluded_extensions", 
        type=str, 
        nargs="*", 
        help="File extensions to exclude"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=2048, 
        help="Maximum tokens per chunk"
    )
    parser.add_argument(
        "--max_chunks_allowed", 
        type=int, 
        default=100, 
        help="Maximum chunks per file"
    )
    parser.add_argument(
        "--max_chunk_characters",
        type=int,
        default=1000000,
        help="Maximum characters per chunk"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't delete the cloned repository after processing"
    )

    args = parser.parse_args()

    # Convert list arguments to sets if provided
    included_extensions = set(args.included_extensions) if args.included_extensions else None
    excluded_extensions = (
        set(args.excluded_extensions)
        if args.excluded_extensions
        else {".png", ".gif", ".bin", ".jpg", ".jpeg", ".mp4", ".csv", ".json"}
    )

    # Initialize and run the chunker
    chunker = CommitSpecificRepoChunker(
        repo_id=args.repo_id,
        commit_hash=args.commit_hash,
        local_dir=args.local_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        included_extensions=included_extensions,
        excluded_extensions=excluded_extensions,
        max_tokens=args.max_tokens,
        max_chunks_allowed=args.max_chunks_allowed,
        max_chunk_characters=args.max_chunk_characters,
    )

    output_path = chunker.chunk(cleanup=not args.no_cleanup)
    print(f"Repository chunking complete!")
    print(f"Output written to: {output_path}")