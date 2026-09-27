#!/usr/bin/env python3
"""
Packaging script for Amazon ML Challenge 2026 submission.

Creates the official submission zip archive with the required structure:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── utils/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import argparse
import os
import zipfile
from pathlib import Path


def create_submission_zip(team_name: str, root_dir: Path, output_tsv_dir: Path, zip_dest_dir: Path):
    zip_dest_dir.mkdir(parents=True, exist_ok=True)
    zip_filename = f"{team_name}_submission.zip"
    zip_path = zip_dest_dir / zip_filename

    print(f"Creating submission package: {zip_path}")
    matching_tsv = output_tsv_dir / "matching_results.tsv"
    candidate_tsv = output_tsv_dir / "candidate_pairs.tsv"
    doc_template = root_dir / "Documentation_template.md"

    if not matching_tsv.is_file():
        raise FileNotFoundError(f"Missing required file: {matching_tsv}")
    if not candidate_tsv.is_file():
        raise FileNotFoundError(f"Missing required file: {candidate_tsv}")
    if not doc_template.is_file():
        raise FileNotFoundError(f"Missing required file: {doc_template}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. output/ folder
        print("Adding output files...")
        zf.write(matching_tsv, arcname="output/matching_results.tsv")
        zf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")

        # 2. Documentation_template.md at root
        print("Adding Documentation_template.md...")
        zf.write(doc_template, arcname="Documentation_template.md")

        # 3. code/business_entity_resolution/ folder
        print("Adding code/business_entity_resolution files...")
        code_prefix = "code/business_entity_resolution"
        
        # Add src/
        for p in (root_dir / "src").rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                rel = p.relative_to(root_dir / "src")
                zf.write(p, arcname=f"{code_prefix}/src/{rel.as_posix()}")

        # Add utils/
        for p in (root_dir / "utils").rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.name != "package_submission.py":
                rel = p.relative_to(root_dir / "utils")
                zf.write(p, arcname=f"{code_prefix}/utils/{rel.as_posix()}")

        # Add README.md & requirements.txt
        if (root_dir / "README.md").is_file():
            zf.write(root_dir / "README.md", arcname=f"{code_prefix}/README.md")
        if (root_dir / "requirements.txt").is_file():
            zf.write(root_dir / "requirements.txt", arcname=f"{code_prefix}/requirements.txt")

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nSuccessfully created {zip_path} ({size_mb:.2f} MB)")
    print("Archive contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename} ({info.file_size:,} bytes)")


def main():
    ap = argparse.ArgumentParser(description="Package Amazon ML Challenge 2026 submission zip")
    ap.add_argument("--team-name", type=str, default="Abbas_Hussain", help="Team name prefix")
    ap.add_argument("--root-dir", type=Path, default=Path("."), help="Project root directory")
    ap.add_argument("--output-tsv-dir", type=Path, default=Path("output/sample_final"), help="Directory containing matching_results.tsv and candidate_pairs.tsv")
    ap.add_argument("--zip-dest-dir", type=Path, default=Path("output"), help="Destination directory for the zip file")
    args = ap.parse_args()

    create_submission_zip(args.team_name, args.root_dir, args.output_tsv_dir, args.zip_dest_dir)


if __name__ == "__main__":
    main()
