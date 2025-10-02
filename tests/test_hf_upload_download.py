"""Tests uploading and downloading files from Hugging Face Hub.

These tests cover the main functionality of the HuggingFace upload and download scripts.
Most tests use mocking to avoid requiring actual HuggingFace Hub access, but some
integration tests can be run with proper authentication.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the scripts directory to the path so we can import the modules
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

try:
    import hf_download_data
    import hf_upload_data
except ImportError as e:
    # Handle case where scripts can't be imported
    print(f"Could not import scripts: {e}")
    hf_upload_data = None  # type: ignore
    hf_download_data = None  # type: ignore


class TestHFUploadData(unittest.TestCase):
    """Test cases for HuggingFace upload functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_files = ["file1.txt", "file2.txt", "subdir/file3.txt"]

        # Create test files
        for file_path in self.test_files:
            full_path = Path(self.test_dir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(f"Content of {file_path}")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_validate_hf_token_valid(self):
        """Test HF token validation with valid token."""
        if hf_upload_data is None:
            self.skipTest("Could not import hf_upload_data")

        # Valid token format
        self.assertTrue(hf_upload_data.validate_hf_token("hf_1234567890abcdef"))

        # Invalid tokens
        self.assertFalse(hf_upload_data.validate_hf_token(""))
        self.assertFalse(
            hf_upload_data.validate_hf_token("YOUR_HUGGINGFACE_TOKEN_HERE")
        )

    def test_get_folder_files(self):
        """Test getting list of files in a folder."""
        if hf_upload_data is None:
            self.skipTest("Could not import hf_upload_data")

        files = hf_upload_data.get_folder_files(self.test_dir)

        # Should find all test files
        self.assertEqual(len(files), 3)
        self.assertIn("file1.txt", files)
        self.assertIn("file2.txt", files)
        self.assertIn(os.path.join("subdir", "file3.txt"), files)

    def test_get_folder_files_nonexistent(self):
        """Test getting files from non-existent folder."""
        if hf_upload_data is None:
            self.skipTest("Could not import hf_upload_data")

        files = hf_upload_data.get_folder_files("/nonexistent/path")
        self.assertEqual(files, [])

    def test_calculate_folder_size(self):
        """Test calculating folder size."""
        if hf_upload_data is None:
            self.skipTest("Could not import hf_upload_data")

        size = hf_upload_data.calculate_folder_size(self.test_dir)

        # Should be greater than 0 since we have files
        self.assertGreater(size, 0)

        # Test with non-existent folder
        size_empty = hf_upload_data.calculate_folder_size("/nonexistent/path")
        self.assertEqual(size_empty, 0)

    def test_format_size(self):
        """Test size formatting."""
        if hf_upload_data is None:
            self.skipTest("Could not import hf_upload_data")

        self.assertEqual(hf_upload_data.format_size(512), "512.0 B")
        self.assertEqual(hf_upload_data.format_size(1024), "1.0 KB")
        self.assertEqual(hf_upload_data.format_size(1024 * 1024), "1.0 MB")

    @patch("hf_upload_data._HF_AVAILABLE", True)
    @patch("hf_upload_data.HfApi")
    def test_upload_folder_dry_run(self, mock_hf_api):
        """Test upload in dry run mode."""
        if hf_upload_data is None:
            self.skipTest("Could not import hf_upload_data")

        # Dry run should not call HuggingFace API
        result = hf_upload_data.upload_folder_to_hf(
            local_path=self.test_dir,
            remote_path="test_remote",
            repo_id="test/repo",
            hf_token="hf_test_token",
            dry_run=True,
        )

        self.assertTrue(result)
        mock_hf_api.assert_not_called()

    @patch("hf_upload_data._HF_AVAILABLE", False)
    def test_upload_folder_no_hf_library(self):
        """Test upload when HuggingFace library is not available."""
        if hf_upload_data is None:
            self.skipTest("Could not import hf_upload_data")

        result = hf_upload_data.upload_folder_to_hf(
            local_path=self.test_dir,
            remote_path="test_remote",
            repo_id="test/repo",
            hf_token="hf_test_token",
        )

        self.assertFalse(result)

    @patch("hf_upload_data._HF_AVAILABLE", True)
    @patch("hf_upload_data.HfApi")
    def test_upload_folder_success(self, mock_hf_api_class):
        """Test successful upload."""
        if hf_upload_data is None:
            self.skipTest("Could not import hf_upload_data")

        # Mock the HfApi instance and its methods
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        result = hf_upload_data.upload_folder_to_hf(
            local_path=self.test_dir,
            remote_path="test_remote",
            repo_id="test/repo",
            hf_token="hf_test_token",
        )

        self.assertTrue(result)
        mock_hf_api_class.assert_called_once_with(token="hf_test_token")
        mock_api.repo_info.assert_called_once()
        mock_api.upload_folder.assert_called_once()

    @patch("hf_upload_data._HF_AVAILABLE", True)
    @patch("hf_upload_data.HfApi")
    def test_upload_folder_create_repo(self, mock_hf_api_class):
        """Test upload with repository creation."""
        if hf_upload_data is None:
            self.skipTest("Could not import hf_upload_data")

        # Mock the HfApi instance
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        # Make repo_info raise an exception to trigger repo creation
        mock_api.repo_info.side_effect = Exception("Repo not found")

        result = hf_upload_data.upload_folder_to_hf(
            local_path=self.test_dir,
            remote_path="test_remote",
            repo_id="test/repo",
            hf_token="hf_test_token",
        )

        self.assertTrue(result)
        mock_api.create_repo.assert_called_once()


class TestHFDownloadData(unittest.TestCase):
    """Test cases for HuggingFace download functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_format_size(self):
        """Test size formatting (same function as upload)."""
        if hf_download_data is None:
            self.skipTest("Could not import hf_download_data")

        self.assertEqual(hf_download_data.format_size(512), "512.0 B")
        self.assertEqual(hf_download_data.format_size(1024), "1.0 KB")
        self.assertEqual(hf_download_data.format_size(1024 * 1024), "1.0 MB")

    @patch("hf_download_data.HfFileSystem")
    def test_get_folder_structure_success(self, mock_hf_fs_class):
        """Test getting folder structure from HF repository."""
        if hf_download_data is None:
            self.skipTest("Could not import hf_download_data")

        # Mock the filesystem response
        mock_fs = MagicMock()
        mock_hf_fs_class.return_value = mock_fs

        mock_files = [
            {"name": "datasets/test/repo/file1.txt", "type": "file", "size": 100},
            {"name": "datasets/test/repo/folder1", "type": "directory"},
            "some_string_entry",  # Should be filtered out
        ]
        mock_fs.ls.return_value = mock_files

        result = hf_download_data.get_folder_structure("test/repo", "")

        # Should filter out non-dict entries
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "datasets/test/repo/file1.txt")
        self.assertEqual(result[1]["name"], "datasets/test/repo/folder1")

    @patch("hf_download_data.HfFileSystem")
    def test_get_folder_structure_error(self, mock_hf_fs_class):
        """Test handling errors when getting folder structure."""
        if hf_download_data is None:
            self.skipTest("Could not import hf_download_data")

        # Mock the filesystem to raise an exception
        mock_fs = MagicMock()
        mock_hf_fs_class.return_value = mock_fs
        mock_fs.ls.side_effect = Exception("Access denied")

        result = hf_download_data.get_folder_structure("test/repo", "")

        self.assertEqual(result, [])

    def test_calculate_download_size(self):
        """Test calculating download size."""
        if hf_download_data is None:
            self.skipTest("Could not import hf_download_data")

        # Mock the get_folder_structure function
        mock_files = [
            {"name": "file1.txt", "type": "file", "size": 100},
            {"name": "file2.txt", "type": "file", "size": 200},
            {"name": "folder1", "type": "directory"},  # No size for directories
            "string_entry",  # Should be filtered out
        ]

        with patch.object(
            hf_download_data, "get_folder_structure", return_value=mock_files
        ):
            size = hf_download_data.calculate_download_size("test/repo", "")
            self.assertEqual(size, 300)  # 100 + 200

    @patch("hf_download_data.HfFileSystem")
    def test_download_folder_dry_run(self, mock_hf_fs_class):
        """Test download in dry run mode."""
        if hf_download_data is None:
            self.skipTest("Could not import hf_download_data")

        # Mock the filesystem
        mock_fs = MagicMock()
        mock_hf_fs_class.return_value = mock_fs

        mock_files = [
            {"name": "datasets/test/repo/file1.txt", "type": "file"},
            {"name": "datasets/test/repo/folder1", "type": "directory"},
        ]
        mock_fs.ls.return_value = mock_files

        result = hf_download_data.download_folder_from_hf(
            repo_id="test/repo",
            hf_folder_path="datasets/test/repo",
            download_dir=self.test_dir,
            dry_run=True,
        )

        self.assertTrue(result)
        # In dry run, no actual downloads should happen
        self.assertEqual(len(os.listdir(self.test_dir)), 0)

    @patch("hf_download_data.HfFileSystem")
    @patch("hf_download_data.hf_hub_download")
    @patch("hf_download_data.shutil.move")
    def test_download_folder_files(self, mock_move, mock_download, mock_hf_fs_class):
        """Test downloading files from HF repository."""
        if hf_download_data is None:
            self.skipTest("Could not import hf_download_data")

        # Mock the filesystem
        mock_fs = MagicMock()
        mock_hf_fs_class.return_value = mock_fs

        mock_files = [{"name": "datasets/test/repo/file1.txt", "type": "file"}]
        mock_fs.ls.return_value = mock_files

        # Mock the download function
        mock_download.return_value = "/tmp/cached_file"

        result = hf_download_data.download_folder_from_hf(
            repo_id="test/repo",
            hf_folder_path="datasets/test/repo",
            download_dir=self.test_dir,
            dry_run=False,
        )

        self.assertTrue(result)
        mock_download.assert_called_once()
        mock_move.assert_called_once()

    @patch("hf_download_data.HfFileSystem")
    def test_download_folder_recursive(self, mock_hf_fs_class):
        """Test recursive download of folders."""
        if hf_download_data is None:
            self.skipTest("Could not import hf_download_data")

        # Mock the filesystem for different calls
        mock_fs = MagicMock()
        mock_hf_fs_class.return_value = mock_fs

        # First call returns a directory, second call returns files in that directory
        mock_fs.ls.side_effect = [
            [{"name": "datasets/test/repo/folder1", "type": "directory"}],
            [{"name": "datasets/test/repo/folder1/file1.txt", "type": "file"}],
        ]

        with (
            patch("hf_download_data.hf_hub_download") as mock_download,
            patch("hf_download_data.shutil.move"),
        ):

            mock_download.return_value = "/tmp/cached_file"

            result = hf_download_data.download_folder_from_hf(
                repo_id="test/repo",
                hf_folder_path="datasets/test/repo",
                download_dir=self.test_dir,
                dry_run=False,
            )

            self.assertTrue(result)
            # Should have made recursive calls
            self.assertEqual(mock_fs.ls.call_count, 2)


class TestMainFunctions(unittest.TestCase):
    """Test the main functions of both scripts."""

    def test_upload_main_validation(self):
        """Test main function argument validation for upload script."""
        if hf_upload_data is None:
            self.skipTest("Could not import hf_upload_data")

        # Test with non-existent folder
        with patch(
            "sys.argv",
            ["hf_upload_data.py", "/nonexistent/path", "--hf_token", "hf_test"],
        ):
            with patch("sys.exit", side_effect=SystemExit) as mock_exit:
                with patch("builtins.print"):  # Suppress error output
                    with self.assertRaises(SystemExit):
                        hf_upload_data.main()
                    mock_exit.assert_called_with(1)

    def test_download_main_validation(self):
        """Test main function argument validation for download script."""
        if hf_download_data is None:
            self.skipTest("Could not import hf_download_data")

        # Test with invalid download directory (existing file)
        with tempfile.NamedTemporaryFile() as temp_file:
            with patch(
                "sys.argv", ["hf_download_data.py", "--download_dir", temp_file.name]
            ):
                # Make sys.exit raise SystemExit so the script actually stops
                with patch("sys.exit", side_effect=SystemExit) as mock_exit:
                    with patch("builtins.print"):  # Suppress error output
                        with self.assertRaises(SystemExit):
                            hf_download_data.main()
                        mock_exit.assert_called_with(1)


class TestIntegration(unittest.TestCase):
    """Integration tests that require actual HuggingFace Hub access.

    These tests are skipped by default but can be run with proper authentication.
    """

    def setUp(self):
        """Set up for integration tests."""
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            self.skipTest("No HF_TOKEN environment variable set")

    @unittest.skip("Requires HuggingFace Hub access and valid token")
    def test_real_upload_download_cycle(self):
        """Test actual upload and download cycle (requires authentication)."""
        if hf_upload_data is None or hf_download_data is None:
            self.skipTest("Could not import required modules")

        # This test would perform actual uploads/downloads
        # Only run with proper authentication and test repository


if __name__ == "__main__":
    unittest.main()
