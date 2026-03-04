"""
Tests for the MOT → YOLO annotation parser.

Validates the conversion logic from MOT Challenge gt.txt format
to normalized YOLO format labels.
"""

import tempfile
from pathlib import Path

import pytest

from src.data.parse_mot import (
    MOTAnnotation,
    SeqInfo,
    group_annotations_by_frame,
    mot_to_yolo,
    parse_gt_file,
    parse_seqinfo,
)


class TestParseSeqinfo:
    """Tests for seqinfo.ini parsing."""

    def test_parse_valid_seqinfo(self, tmp_path):
        """Test parsing a well-formed seqinfo.ini file."""
        seqinfo = tmp_path / "seqinfo.ini"
        seqinfo.write_text(
            "[Sequence]\n"
            "name=v_test_c001\n"
            "imDir=img1\n"
            "frameRate=25\n"
            "seqLength=100\n"
            "imWidth=1280\n"
            "imHeight=720\n"
            "imExt=.jpg\n"
        )

        info = parse_seqinfo(seqinfo)

        assert info.name == "v_test_c001"
        assert info.img_dir == "img1"
        assert info.frame_rate == 25
        assert info.seq_length == 100
        assert info.img_width == 1280
        assert info.img_height == 720
        assert info.img_ext == ".jpg"

    def test_seqinfo_not_found(self, tmp_path):
        """Test that missing seqinfo.ini raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_seqinfo(tmp_path / "nonexistent.ini")


class TestParseGtFile:
    """Tests for gt.txt parsing."""

    def test_parse_valid_gt(self, tmp_path):
        """Test parsing a valid gt.txt with multiple lines."""
        gt_file = tmp_path / "gt.txt"
        gt_file.write_text(
            "1,1,100,200,50,120,1,1,1\n"
            "1,2,300,150,60,130,1,1,1\n"
            "2,1,110,205,50,120,1,1,1\n"
        )

        annotations = parse_gt_file(gt_file)

        assert len(annotations) == 3
        assert annotations[0].frame == 1
        assert annotations[0].track_id == 1
        assert annotations[0].bb_left == 100
        assert annotations[0].bb_top == 200
        assert annotations[0].bb_width == 50
        assert annotations[0].bb_height == 120

    def test_parse_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        gt_file = tmp_path / "gt.txt"
        gt_file.write_text(
            "1,1,100,200,50,120,1,1,1\n"
            "\n"
            "2,1,110,205,50,120,1,1,1\n"
            "\n"
        )

        annotations = parse_gt_file(gt_file)
        assert len(annotations) == 2

    def test_parse_malformed_line(self, tmp_path):
        """Test that malformed lines are skipped with a warning."""
        gt_file = tmp_path / "gt.txt"
        gt_file.write_text(
            "1,1,100,200,50,120,1,1,1\n"
            "bad,line\n"
            "2,1,110,205,50,120,1,1,1\n"
        )

        annotations = parse_gt_file(gt_file)
        assert len(annotations) == 2

    def test_gt_file_not_found(self, tmp_path):
        """Test that missing gt.txt raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_gt_file(tmp_path / "nonexistent.txt")


class TestGroupAnnotationsByFrame:
    """Tests for frame grouping."""

    def test_group_correctly(self):
        """Test that annotations are grouped by frame number."""
        annotations = [
            MOTAnnotation(1, 1, 100, 200, 50, 120, 1, 1, 1),
            MOTAnnotation(1, 2, 300, 150, 60, 130, 1, 1, 1),
            MOTAnnotation(2, 1, 110, 205, 50, 120, 1, 1, 1),
            MOTAnnotation(3, 1, 120, 210, 50, 120, 1, 1, 1),
            MOTAnnotation(3, 2, 310, 155, 60, 130, 1, 1, 1),
        ]

        groups = group_annotations_by_frame(annotations)

        assert len(groups) == 3
        assert len(groups[1]) == 2
        assert len(groups[2]) == 1
        assert len(groups[3]) == 2

    def test_empty_list(self):
        """Test grouping an empty annotation list."""
        groups = group_annotations_by_frame([])
        assert groups == {}


class TestMotToYolo:
    """Tests for MOT → YOLO format conversion."""

    def test_basic_conversion(self):
        """Test basic bounding box conversion to normalized YOLO format."""
        ann = MOTAnnotation(
            frame=1, track_id=1,
            bb_left=100, bb_top=200, bb_width=50, bb_height=120,
            conf=1, cls=1, visibility=1,
        )

        yolo_line = mot_to_yolo(ann, img_width=1280, img_height=720)

        parts = yolo_line.split()
        assert parts[0] == "0"  # class_id

        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        # Expected: cx = (100 + 25) / 1280 = 0.09765625
        assert abs(cx - (100 + 25) / 1280) < 1e-5
        # Expected: cy = (200 + 60) / 720 = 0.36111...
        assert abs(cy - (200 + 60) / 720) < 1e-5
        # Expected: w = 50 / 1280 = 0.0390625
        assert abs(w - 50 / 1280) < 1e-5
        # Expected: h = 120 / 720 = 0.16666...
        assert abs(h - 120 / 720) < 1e-5

    def test_clamping(self):
        """Test that values exceeding image bounds are clamped to [0, 1]."""
        ann = MOTAnnotation(
            frame=1, track_id=1,
            bb_left=1250, bb_top=700, bb_width=100, bb_height=100,
            conf=1, cls=1, visibility=1,
        )

        yolo_line = mot_to_yolo(ann, img_width=1280, img_height=720)
        parts = yolo_line.split()

        # All values should be clamped to [0, 1]
        for val_str in parts[1:]:
            val = float(val_str)
            assert 0.0 <= val <= 1.0, f"Value {val} out of [0, 1] range"

    def test_full_frame_box(self):
        """Test a bounding box covering the entire frame."""
        ann = MOTAnnotation(
            frame=1, track_id=1,
            bb_left=0, bb_top=0, bb_width=1280, bb_height=720,
            conf=1, cls=1, visibility=1,
        )

        yolo_line = mot_to_yolo(ann, img_width=1280, img_height=720)
        parts = yolo_line.split()

        assert abs(float(parts[1]) - 0.5) < 1e-5  # cx = 0.5
        assert abs(float(parts[2]) - 0.5) < 1e-5  # cy = 0.5
        assert abs(float(parts[3]) - 1.0) < 1e-5  # w = 1.0
        assert abs(float(parts[4]) - 1.0) < 1e-5  # h = 1.0
