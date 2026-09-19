"""All prompt templates used by the PDF bookmarks tool."""


class TOCDetectionPrompts:
    """Prompts for TOC (Table of Contents) page detection."""

    IS_TOC_PAGE = (
        "This page is from a book. Determine if it is a table of contents. "
        "A table of contents MUST have BOTH section titles AND explicit page numbers for each entry. "
        "Look for clear page number annotations (usually aligned on the right) next to each section title. "
        "If the page has section listings but NO page numbers, it is NOT a table of contents. "
        "The layout is typically structured with entries in a list format, often with hierarchical indentation. "
        "Answer ONLY 'yes' or 'no'."
    )


class TOCExtractionPrompts:
    """Prompts for extracting entries from TOC pages."""

    EXTRACT_FIRST_ARABIC_ENTRY = (
        "From this table of contents page, find the very first entry whose page number is an Arabic numeral "
        "(like 1, 3, 15 — NOT i, ii, iii, v, x, etc.). "
        "Extract the COMPLETE title text exactly as shown, including all characters. "
        "Return EXACTLY in the format: 'page_number, entry_text'. "
        "If no such entry exists on this page, return 'none, none'."
    )

    EXTRACT_VERIFICATION_ENTRIES = (
        "From this table of contents page, the first Arabic entry is on page {first_entry_page}. "
        "Find 2-3 OTHER entries that are FAR from this first entry (preferably near the middle and end of the page). "
        "For each entry, extract the page number and COMPLETE title. "
        "Return ONLY in this exact format (one per line):\n"
        "page_number, title\n"
        "page_number, title\n"
        "page_number, title\n\n"
        "Example:\n"
        "50, Chapter 3 Advanced Topics\n"
        "120, Appendix A Reference\n"
        "Do not include the first entry (page {first_entry_page}). Only return entries with Arabic numeral page numbers."
    )


class ContentVerificationPrompts:
    """Prompts for verifying content on pages."""

    PAGE_CONTAINS_CONTENT = (
        "Does this page contain the main body content of the section titled '{target_text}'? "
        "Look for key terms from the title on this page. "
        "Ignore headers, footers, or references. Only say 'yes' if this appears to be the actual start of that section. "
        "Answer ONLY 'yes' or 'no'."
    )

    VERIFY_OFFSET_MATCH = (
        "You are verifying a page offset calculation.\n\n"
        "According to the table of contents, the section '{expected_title}' should be on page {expected_page}.\n\n"
        "Does this page appear to be the correct location for '{expected_title}'?\n\n"
        "Check for:\n"
        "- Key terms from the title ('{expected_title}') appearing on this page\n"
        "- This being the start of that section (not a middle/end page)\n"
        "- The content being reasonably consistent with the title\n\n"
        "Be somewhat lenient - small variations in exact wording are acceptable.\n"
        "Answer ONLY 'yes' or 'no'."
    )


class BookmarkGenerationPrompts:
    """Prompts for generating bookmarks from TOC pages."""

    FIRST_PAGE_PROMPT = """
You are analyzing a book's table of contents page.

Your task: Extract ALL bookmark entries from this page.

Rules:
- ONLY include entries that correspond to Arabic-numbered pages (e.g., 1, 2, 15).
- EXCLUDE any entry whose page number is a Roman numeral (e.g., i, ii, iii, vi, xii).
- For each valid entry, output:
BookmarkBegin
BookmarkTitle: [Exact title from the table of contents]
BookmarkLevel: [1 for top-level, 2 for subsection, etc.]
BookmarkPageNumber: [Original page number as Arabic integer]

Example of expected output format:
BookmarkBegin
BookmarkTitle: Chapter 1: Introduction
BookmarkLevel: 1
BookmarkPageNumber: 5
BookmarkBegin
BookmarkTitle: 1.1 Background
BookmarkLevel: 2
BookmarkPageNumber: 7

Output ONLY the bookmark entries from this page, no explanation.
    """.strip()



class BookmarkRefinementPrompts:
    """Prompts for refining generated bookmarks."""

    REFINE_BOOKMARKS = """
You are given a pdftk bookmark format output that was extracted from a book's table of contents.

Your task: Review and fix any errors in the bookmark structure.

Rules to follow:
- Ensure proper nesting hierarchy (BookmarkLevel must be consistent)
- Check that page numbers are reasonable (positive integers, generally increasing)
- Fix any formatting issues
- Remove any duplicate entries
- The output must remain in valid pdftk format

Here is the bookmark text to review:
```
{bookmark_text}
```

Use replace_text tool calls to make only the necessary local corrections to the current text.
Each old_text must match exactly once; include surrounding text when necessary.
Edits apply sequentially, so later edits must refer to the text after earlier edits.
Never output or replace the entire document. Preserve unaffected titles and page numbers.
Call finish_refinement when done, including when no edits are needed.
    """.strip()
