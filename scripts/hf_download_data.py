#!/usr/bin/env python3
"""
Script to download data from HuggingFace datasets repository.

This script downloads specified folders from a HuggingFace datasets repository,
maintaining the folder structure locally.

Example usage:
    # Download specific folder
    python hf_download_data.py --download_dir ./data --remote_path experiments/run1
    
    # Download with custom repository
    python hf_download_data.py --download_dir ./data --remote_path data --repo_id org/dataset
    
    # Download entire repository
    python hf_download_data.py --download_dir ./data --remote_path ""
    
    # Dry run to see what would be downloaded
    python hf_download_data.py --download_dir ./data --remote_path experiments --dry_run
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List

from huggingface_hub import HfFileSystem, hf_hub_download

# Repository configuration
REPO_ID = "vaibhavsaxena11/prbench-data"  # Change this to your desired repo ID


def get_folder_structure(repo_id: str, remote_path: str = "") -> List[dict]:
    """
    Get the structure of files and folders in a HuggingFace repository path.
    
    Args:
        repo_id (str): Repository ID
        remote_path (str): Path in repository (empty for root)
        
    Returns:
        List[dict]: List of file/folder information
    """
    fs = HfFileSystem()
    
    # Construct the full path
    if remote_path:
        hf_path = f"datasets/{repo_id}/{remote_path}"
    else:
        hf_path = f"datasets/{repo_id}"
    
    try:
        files = fs.ls(hf_path, detail=True)
        return files
    except Exception as e:
        print(f"Error accessing {hf_path}: {e}")
        return []


def download_folder_from_hf(
    repo_id: str,
    hf_folder_path: str,
    download_dir: str,
    dry_run: bool = False,
    verbose: bool = False
) -> bool:
    """
    Download all contents of a folder from HuggingFace repository.
    This function is recursive, downloading all subfolders and files.

    Args:
        repo_id (str): Repository ID
        hf_folder_path (str): Path to the folder in HuggingFace
        download_dir (str): Directory to download the folder to
        dry_run (bool): If True, only show what would be downloaded
        verbose (bool): If True, show detailed progress
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        fs = HfFileSystem()

        # List all files/folders in the repo folder
        files = fs.ls(hf_folder_path, detail=True)
        
        if dry_run:
            print(f"[DRY RUN] Would download from {hf_folder_path} to {download_dir}")
            print(f"  Found {len(files)} items:")
            for file in files:
                basename = os.path.basename(file["name"])
                file_type = "📁" if file["type"] == "directory" else "📄"
                print(f"    {file_type} {basename}")
            return True

        if verbose:
            print(f"📁 Downloading from {hf_folder_path}")
            print(f"   Target: {download_dir}")

        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)

        for file in files:
            filename = file["name"]  # Full path in HF
            basename = os.path.basename(filename)  # Just the file/folder name

            if file["type"] == "directory":
                # Recursively download folder
                download_subdir = os.path.join(download_dir, basename)
                if verbose:
                    print(f"  📁 Entering folder: {basename}")
                
                success = download_folder_from_hf(
                    repo_id=repo_id,
                    hf_folder_path=filename,
                    download_dir=download_subdir,
                    dry_run=dry_run,
                    verbose=verbose
                )
                if not success:
                    return False
            else:
                # Download file
                if verbose:
                    print(f"  📄 Downloading: {basename}")
                
                # Create temporary directory for download
                tmp_dir = os.path.join(download_dir, "tmp")
                
                # Get the relative path within the repository
                filename_in_repo = filename[len(f"datasets/{repo_id}/"):]
                
                try:
                    fpath = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename_in_repo,
                        repo_type="dataset",
                        cache_dir=tmp_dir,
                    )
                    
                    # Move downloaded file to the target directory
                    target_path = os.path.join(download_dir, basename)
                    shutil.move(os.path.realpath(fpath), target_path)
                    
                    # Clean up temporary directory
                    if os.path.exists(tmp_dir):
                        shutil.rmtree(tmp_dir)
                        
                except Exception as e:
                    print(f"    ❌ Error downloading {basename}: {e}")
                    # Clean up temporary directory on error
                    if os.path.exists(tmp_dir):
                        shutil.rmtree(tmp_dir)
                    return False

        return True
        
    except Exception as e:
        print(f"❌ Error downloading from {hf_folder_path}: {e}")
        return False


def calculate_download_size(repo_id: str, remote_path: str = "") -> int:
    """
    Estimate the total size of files to be downloaded.
    
    Args:
        repo_id (str): Repository ID
        remote_path (str): Path in repository
        
    Returns:
        int: Total size in bytes (approximate)
    """
    # Note: This is a basic implementation
    # HuggingFace doesn't always provide file sizes easily
    files = get_folder_structure(repo_id, remote_path)
    total_size = 0
    
    for file in files:
        if file["type"] == "file" and "size" in file:
            total_size += file.get("size", 0)
    
    return total_size


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="Download data from HuggingFace datasets repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific folder
  python hf_download_data.py --download_dir ./data --remote_path experiments/run1
  
  # Download from custom repository
  python hf_download_data.py --download_dir ./data --remote_path data --repo_id org/dataset
  
  # Download entire repository
  python hf_download_data.py --download_dir ./data --remote_path ""
  
  # Dry run to see what would be downloaded
  python hf_download_data.py --download_dir ./data --remote_path experiments --dry_run --verbose
        """
    )
    
    parser.add_argument(
        "--download_dir",
        type=str,
        required=True,
        help="Directory to download data to"
    )
    
    parser.add_argument(
        "--remote_path",
        type=str,
        default="",
        help="Path in repository to download (empty for entire repository)"
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        default=REPO_ID,
        help=f"Repository ID on HuggingFace Hub (default: {REPO_ID})"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only show what would be downloaded without actually downloading"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress information"
    )
    
    args = parser.parse_args()
    
    # Validate download directory
    download_dir = Path(args.download_dir)
    if download_dir.exists() and not download_dir.is_dir():
        print(f"Error: Download path exists but is not a directory: {args.download_dir}")
        exit(1)
    
    # Create download directory if it doesn't exist
    if not args.dry_run:
        download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Repository: {args.repo_id}")
    print(f"Remote path: {args.remote_path if args.remote_path else '(entire repository)'}")
    print(f"Download directory: {args.download_dir}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual downloads will be performed")
    
    print(f"{'='*80}")
    
    # Construct the HuggingFace path
    if args.remote_path:
        hf_folder_path = f"datasets/{args.repo_id}/{args.remote_path}"
        target_dir = os.path.join(args.download_dir, os.path.basename(args.remote_path))
    else:
        hf_folder_path = f"datasets/{args.repo_id}"
        target_dir = args.download_dir
    
    # Check if the path exists in the repository
    files = get_folder_structure(args.repo_id, args.remote_path)
    if not files:
        print(f"❌ No files found at path: {args.remote_path}")
        print(f"   Repository: {args.repo_id}")
        exit(1)
    
    # Download the data
    success = download_folder_from_hf(
        repo_id=args.repo_id,
        hf_folder_path=hf_folder_path,
        download_dir=target_dir,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    print(f"{'='*80}")
    
    if success:
        if args.dry_run:
            print("✅ Dry run completed successfully")
        else:
            print("🎉 Download completed successfully!")
            print(f"   Data saved to: {target_dir}")
    else:
        print("❌ Download failed")
        exit(1)


if __name__ == "__main__":
    main()