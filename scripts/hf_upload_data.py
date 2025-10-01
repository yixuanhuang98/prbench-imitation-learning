#!/usr/bin/env python3
"""
Script to upload a folder to HuggingFace datasets repository.

This script uploads a specified folder to a HuggingFace datasets repository,
maintaining the folder structure on HuggingFace.

Example usage:
    # Upload a folder
    python hf_upload_data.py /path/to/folder --hf_token YOUR_TOKEN
    
    # Upload with custom repository
    python hf_upload_data.py /path/to/folder --hf_token YOUR_TOKEN --repo_id your-org/your-dataset
    
    # Dry run to see what would be uploaded
    python hf_upload_data.py /path/to/folder --hf_token YOUR_TOKEN --dry_run
    
    # Upload to specific path in repository
    python hf_upload_data.py /path/to/folder --hf_token YOUR_TOKEN --remote_path data/experiments
"""

import os
import argparse
from pathlib import Path
from typing import List
import time

# Repository configuration
REPO_ID = "vaibhavsaxena11/prbench-data"  # Change this to your desired repo ID
REPO_TYPE = "dataset"


def validate_hf_token(token: str) -> bool:
    """
    Validate HuggingFace token format and check if it's not a placeholder.
    
    Args:
        token (str): HuggingFace token
        
    Returns:
        bool: True if token appears valid, False otherwise
    """
    if not token or token == "YOUR_HUGGINGFACE_TOKEN_HERE":
        return False
    
    # Basic validation - HF tokens typically start with 'hf_'
    if not token.startswith('hf_'):
        print("Warning: HuggingFace tokens typically start with 'hf_'")
    
    return True


def get_folder_files(folder_path: str) -> List[str]:
    """
    Get list of all files in a folder recursively.
    
    Args:
        folder_path (str): Path to folder
        
    Returns:
        List[str]: List of file paths relative to folder_path
    """
    files = []
    
    if not os.path.exists(folder_path):
        return files
    
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            full_path = os.path.join(root, filename)
            relative_path = os.path.relpath(full_path, folder_path)
            files.append(relative_path)
    
    return files


def calculate_folder_size(folder_path: str) -> int:
    """
    Calculate total size of folder in bytes.
    
    Args:
        folder_path (str): Path to folder
        
    Returns:
        int: Total size in bytes
    """
    total_size = 0
    
    if not os.path.exists(folder_path):
        return total_size
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, IOError):
                pass
    
    return total_size


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def upload_folder_to_hf(
    local_path: str, 
    remote_path: str, 
    repo_id: str,
    hf_token: str, 
    dry_run: bool = False, 
    verbose: bool = False
) -> bool:
    """
    Upload a folder to HuggingFace datasets.
    
    Args:
        local_path (str): Local path to folder
        remote_path (str): Remote path in HuggingFace repository
        repo_id (str): Repository ID
        hf_token (str): HuggingFace token
        dry_run (bool): If True, only show what would be uploaded
        verbose (bool): If True, show detailed progress
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from huggingface_hub import HfApi
        
        files = get_folder_files(local_path)
        size = calculate_folder_size(local_path)
        
        if dry_run:
            print(f"[DRY RUN] Would upload {len(files)} files ({format_size(size)})")
            print(f"  Local path: {local_path}")
            print(f"  Remote path: {remote_path}")
            print(f"  Repository: {repo_id}")
            if verbose:
                print("  Files to upload:")
                for file in sorted(files):
                    print(f"    {file}")
            return True
        
        # Initialize HuggingFace API
        api = HfApi(token=hf_token)
        
        print(f"📁 Uploading {len(files)} files ({format_size(size)}) to {repo_id}")
        print(f"   Local: {local_path}")
        print(f"   Remote: {remote_path}")
        
        # Check if repository exists, create if it doesn't
        try:
            api.repo_info(repo_id=repo_id, repo_type=REPO_TYPE)
            if verbose:
                print(f"✓ Repository '{repo_id}' exists")
        except Exception:
            print(f"📝 Creating new repository '{repo_id}'...")
            api.create_repo(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                exist_ok=True
            )
        
        # Upload folder
        api.upload_folder(
            folder_path=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            commit_message=f"Upload {Path(local_path).name} to {remote_path}",
        )
        
        print(f"✅ Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
        
        return True
        
    except ImportError:
        print("❌ Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Error uploading to {remote_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload a folder to HuggingFace datasets repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic upload
  python hf_upload_data.py /path/to/folder --hf_token YOUR_TOKEN
  
  # Upload to custom repository
  python hf_upload_data.py /path/to/folder --hf_token YOUR_TOKEN --repo_id org/dataset
  
  # Upload to specific path in repository
  python hf_upload_data.py /path/to/folder --hf_token YOUR_TOKEN --remote_path experiments/run1
  
  # Dry run
  python hf_upload_data.py /path/to/folder --hf_token YOUR_TOKEN --dry_run --verbose
        """
    )
    
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder to upload"
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="HuggingFace token for authentication"
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        default=REPO_ID,
        help=f"Repository ID on HuggingFace Hub (default: {REPO_ID})"
    )
    
    parser.add_argument(
        "--remote_path",
        type=str,
        default="",
        help="Remote path in repository (default: root of repository)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only show what would be uploaded without actually uploading"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress information"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force upload even if token validation fails"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.folder_path):
        print(f"Error: Folder does not exist: {args.folder_path}")
        exit(1)
    
    if not os.path.isdir(args.folder_path):
        print(f"Error: Path is not a directory: {args.folder_path}")
        exit(1)
    
    if not args.dry_run and not validate_hf_token(args.hf_token) and not args.force:
        print("Error: Invalid or missing HuggingFace token.")
        print("Please provide a valid token with --hf_token")
        print("Get your token from: https://huggingface.co/settings/tokens")
        print("Use --force to bypass token validation (not recommended)")
        exit(1)
    
    # Set remote path - if not specified, use the folder name
    if not args.remote_path:
        folder_name = Path(args.folder_path).name
        remote_path = folder_name
    else:
        remote_path = args.remote_path
    
    print(f"Folder to upload: {args.folder_path}")
    print(f"HuggingFace repository: {args.repo_id}")
    print(f"Remote path: {remote_path}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual uploads will be performed")
    
    print(f"{'='*80}")
    
    # Upload the folder
    success = upload_folder_to_hf(
        local_path=args.folder_path,
        remote_path=remote_path,
        repo_id=args.repo_id,
        hf_token=args.hf_token,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    print(f"{'='*80}")
    
    if success:
        if args.dry_run:
            print("✅ Dry run completed successfully")
        else:
            print("🎉 Upload completed successfully!")
    else:
        print("❌ Upload failed")
        exit(1)


if __name__ == "__main__":
    main()