"""
Tests for CLAUDE.md — the AI-assistant guidance document added at the repo root.

This repository is documentation-only (README.md + docs/), so there is no
application source code to exercise. These tests instead validate the
*structure and content* of CLAUDE.md itself:

- required sections/headings are present
- the "Contents" table only references files that actually exist
- the job-ladder table (L0-L4) is well-formed and consistent with README.md
- the architecture diagram is consistent with README.md
- the security-baseline summary mentions the key safeguards
- the "Related Repos" list contains the expected repositories
- general markdown hygiene (non-empty, well-formed tables, no leftover
  placeholder markers)

Run with:
    python3 -m unittest tests/test_claude_md.py -v
"""
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAUDE_MD_PATH = REPO_ROOT / "CLAUDE.md"
README_PATH = REPO_ROOT / "README.md"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_markdown_table(text: str, header_pattern: str):
    """Locate a markdown table whose header line matches `header_pattern`
    and return (header_cells, [row_cells, ...])."""
    lines = text.splitlines()
    header_re = re.compile(header_pattern)
    start = None
    for i, line in enumerate(lines):
        if header_re.match(line.strip()):
            start = i
            break
    if start is None:
        raise AssertionError(f"No table found matching header pattern: {header_pattern!r}")

    def split_row(line: str):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        return cells

    header_cells = split_row(lines[start])
    # lines[start + 1] must be the separator row, e.g. |---|---|---|
    separator = lines[start + 1].strip()
    assert re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?$", separator), (
        f"Expected a markdown table separator row after header, got: {separator!r}"
    )

    rows = []
    idx = start + 2
    while idx < len(lines) and lines[idx].strip().startswith("|"):
        rows.append(split_row(lines[idx]))
        idx += 1

    return header_cells, rows


def extract_fenced_code_block(text: str, after_marker: str) -> str:
    """Return the contents of the first fenced code block that appears
    after the given marker substring."""
    marker_idx = text.index(after_marker)
    remainder = text[marker_idx:]
    match = re.search(r"```[^\n]*\n(.*?)```", remainder, re.DOTALL)
    assert match, f"No fenced code block found after marker: {after_marker!r}"
    return match.group(1)


def normalize_whitespace(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()


class ClaudeMdFileTests(unittest.TestCase):
    """Basic sanity checks on the file itself."""

    @classmethod
    def setUpClass(cls):
        cls.text = read_text(CLAUDE_MD_PATH)

    def test_file_exists(self):
        self.assertTrue(CLAUDE_MD_PATH.is_file(), "CLAUDE.md should exist at the repo root")

    def test_file_not_empty(self):
        self.assertTrue(len(self.text.strip()) > 0, "CLAUDE.md should not be empty")

    def test_starts_with_top_level_title(self):
        self.assertTrue(
            self.text.startswith("# CLAUDE.md"),
            "CLAUDE.md should start with a top-level '# CLAUDE.md' heading",
        )

    def test_no_placeholder_markers(self):
        for marker in ("TODO", "FIXME", "XXX", "TBD", "Lorem ipsum"):
            self.assertNotIn(
                marker, self.text, f"Found leftover placeholder marker {marker!r} in CLAUDE.md"
            )

    def test_required_sections_present(self):
        expected_headings = [
            "## Project Overview",
            "## Contents",
            "## The Job Ladder (L0–L4)",
            "## System Architecture (documented here, implemented in the code repos)",
            "## Working On This Repo",
            "## Related Repos (where the code lives)",
        ]
        for heading in expected_headings:
            self.assertIn(heading, self.text, f"Missing expected heading: {heading!r}")

    def test_declares_docs_only_repo(self):
        self.assertIn(
            "docs-only repo",
            self.text,
            "CLAUDE.md should describe the repo as docs-only",
        )


class ContentsTableTests(unittest.TestCase):
    """The 'Contents' table should only reference files that exist."""

    @classmethod
    def setUpClass(cls):
        cls.text = read_text(CLAUDE_MD_PATH)
        cls.header, cls.rows = extract_markdown_table(cls.text, r"^\|\s*Path\s*\|\s*Purpose\s*\|")

    def test_table_header(self):
        self.assertEqual(self.header, ["Path", "Purpose"])

    def test_table_has_rows(self):
        self.assertGreaterEqual(len(self.rows), 2, "Contents table should list at least README.md and docs/TESTING.md")

    def test_referenced_paths_exist(self):
        path_cell_re = re.compile(r"`([^`]+)`")
        referenced_paths = []
        for row in self.rows:
            path_cell = row[0]
            match = path_cell_re.search(path_cell)
            self.assertIsNotNone(match, f"Path cell should be wrapped in backticks: {path_cell!r}")
            referenced_paths.append(match.group(1))

        self.assertIn("README.md", referenced_paths)
        self.assertIn("docs/TESTING.md", referenced_paths)

        for rel_path in referenced_paths:
            full_path = REPO_ROOT / rel_path
            self.assertTrue(
                full_path.is_file(), f"Contents table references a path that does not exist: {rel_path!r}"
            )

    def test_every_row_has_a_purpose(self):
        for row in self.rows:
            self.assertEqual(len(row), 2, f"Expected 2 columns per row, got: {row!r}")
            self.assertTrue(row[1], f"Purpose column should not be empty for row: {row!r}")


class JobLadderTableTests(unittest.TestCase):
    """The L0-L4 job ladder table should be well-formed and match README.md."""

    HEADER_PATTERN = r"^\|\s*Level\s*\|\s*Actor\s*\|\s*Behaviour\s*\|"

    @classmethod
    def setUpClass(cls):
        cls.claude_text = read_text(CLAUDE_MD_PATH)
        cls.readme_text = read_text(README_PATH)
        cls.header, cls.rows = extract_markdown_table(cls.claude_text, cls.HEADER_PATTERN)

    def test_table_header(self):
        self.assertEqual(self.header, ["Level", "Actor", "Behaviour"])

    def test_has_exactly_five_levels(self):
        self.assertEqual(len(self.rows), 5, "Job ladder should define exactly 5 levels (L0-L4)")

    def test_levels_are_l0_through_l4_in_order(self):
        levels = [row[0] for row in self.rows]
        self.assertEqual(levels, ["L0", "L1", "L2", "L3", "L4"])

    def test_l0_through_l3_are_agent_actor(self):
        for row in self.rows[:4]:
            level, actor, _ = row
            self.assertEqual(actor, "Agent", f"{level} should be actor 'Agent'")

    def test_l4_is_human_actor(self):
        level, actor, _ = self.rows[-1]
        self.assertEqual(level, "L4")
        self.assertEqual(actor, "Human")

    def test_each_row_has_non_empty_behaviour(self):
        for row in self.rows:
            self.assertTrue(row[2], f"Behaviour column should not be empty for row: {row!r}")

    def test_matches_readme_job_ladder_table(self):
        _, readme_rows = extract_markdown_table(self.readme_text, self.HEADER_PATTERN)
        self.assertEqual(
            self.rows,
            readme_rows,
            "The job ladder table in CLAUDE.md should stay in sync with the one in README.md",
        )


class ArchitectureDiagramTests(unittest.TestCase):
    """The architecture diagram should be consistent with README.md's diagram."""

    @classmethod
    def setUpClass(cls):
        cls.claude_text = read_text(CLAUDE_MD_PATH)
        cls.readme_text = read_text(README_PATH)

    def test_diagram_contains_expected_components(self):
        diagram = extract_fenced_code_block(self.claude_text, "## System Architecture")
        for component in (
            "React 19 Cockpit",
            "FastAPI",
            "SQLAlchemy async",
            "n8n",
            "MCP server",
        ):
            self.assertIn(component, diagram, f"Architecture diagram missing component: {component!r}")

    def test_diagram_matches_readme_diagram(self):
        claude_diagram = extract_fenced_code_block(self.claude_text, "## System Architecture")
        readme_diagram = extract_fenced_code_block(self.readme_text, "## Current architecture")

        claude_lines = [normalize_whitespace(l) for l in claude_diagram.splitlines() if l.strip()]
        readme_lines = [normalize_whitespace(l) for l in readme_diagram.splitlines() if l.strip()]

        self.assertEqual(
            claude_lines,
            readme_lines,
            "Architecture diagram in CLAUDE.md should match README.md (ignoring whitespace differences)",
        )


class SecurityBaselineTests(unittest.TestCase):
    """The condensed security-baseline summary should mention the key safeguards."""

    @classmethod
    def setUpClass(cls):
        cls.text = read_text(CLAUDE_MD_PATH)

    def test_mentions_all_key_safeguards(self):
        expected_terms = [
            "JWT",
            "brute-force lockout",
            "shell_exec",
            "allowlist",
            "shell=False",
            "WebSocket",
            "CORS",
            "wildcard",
            "SECRET_KEY",
            "AB_PASSWORD",
        ]
        for term in expected_terms:
            self.assertIn(term, self.text, f"Security baseline summary should mention {term!r}")


class RelatedReposTests(unittest.TestCase):
    """The 'Related Repos' list should reference the expected set of repos."""

    @classmethod
    def setUpClass(cls):
        section_start = read_text(CLAUDE_MD_PATH).index("## Related Repos")
        cls.section_text = read_text(CLAUDE_MD_PATH)[section_start:]
        cls.bullets = [
            line.strip()[2:].strip()
            for line in cls.section_text.splitlines()
            if line.strip().startswith("- ")
        ]

    def test_has_bullets(self):
        self.assertGreater(len(self.bullets), 0, "Related Repos section should have bullet points")

    def test_lists_expected_repos(self):
        joined = "\n".join(self.bullets)
        for repo in (
            "Aurora-AI-Agency",
            "autoborosai-dashboard",
            "autoborosai.com",
            "autoboros-backend",
            "autoboros-cockpit",
        ):
            self.assertIn(repo, joined, f"Related Repos section should mention {repo!r}")

    def test_canonical_implementation_is_flagged(self):
        joined = "\n".join(self.bullets)
        self.assertIn(
            "Canonical implementation",
            joined,
            "Related Repos section should identify which repo is the canonical implementation",
        )


class MarkdownTableHygieneTests(unittest.TestCase):
    """General well-formedness checks for every markdown table in CLAUDE.md."""

    @classmethod
    def setUpClass(cls):
        cls.text = read_text(CLAUDE_MD_PATH)

    def test_all_pipe_rows_have_consistent_column_counts(self):
        lines = self.text.splitlines()
        current_header_cols = None
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("|"):
                current_header_cols = None
                continue
            cols = [c.strip() for c in stripped.strip("|").split("|")]
            if current_header_cols is None:
                current_header_cols = len(cols)
                continue
            # Separator rows (e.g. |---|---|) are allowed to just repeat dashes
            self.assertEqual(
                len(cols),
                current_header_cols,
                f"Table row has inconsistent column count: {line!r}",
            )


if __name__ == "__main__":
    unittest.main()