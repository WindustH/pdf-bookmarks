"""Vision LLM client for processing PDF TOC pages."""

import sys
import json
import re
import time
import math
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from openai import OpenAI, APITimeoutError
from openai.types.chat import ChatCompletionMessage

from pdf_bookmarks.core.image import PDFImageProcessor
from pdf_bookmarks.utils import Colors, Log, clean_llm_response
from pdf_bookmarks.prompts import (
    TOCDetectionPrompts,
    TOCExtractionPrompts,
    ContentVerificationPrompts,
    BookmarkRefinementPrompts,
)


class VisionLLMClient:
    """Handles all interactions with the vision language model."""

    def __init__(self, api_key: str, base_url: str, vision_model: str, text_model: str, refine_timeout: float = 600):
        if not math.isfinite(refine_timeout) or refine_timeout <= 0:
            raise ValueError("REFINE_TIMEOUT must be a positive finite number")
        self.refine_timeout = refine_timeout
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.vision_model = vision_model
        self.text_model = text_model

    def _send_vision_request(self, images: List[str], prompt: str, stream: bool = True, timeout: float = 120, display: bool = True) -> str:
        """Send a request to the vision LLM with images and prompt."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        for base64_image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/webp;base64,{base64_image}",
                        "detail": "high",
                    },
                }
            )

        return self._complete_text(
            self.vision_model, [{"role": "user", "content": content}],
            stream=stream, timeout=timeout, display=display,
        )

    def _complete_text(self, model, messages, *, stream=True, timeout=60, display=True):
        """Continue length-limited responses without changing their exact text."""
        initial_messages = list(messages)
        full_content = ""
        for attempt in range(21):
            response = self.client.chat.completions.create(
                model=model, messages=messages, stream=stream, timeout=timeout,
            )
            if stream:
                content, reason = self._process_streaming_response(response, display=display)
            else:
                content = response.choices[0].message.content or ""
                reason = response.choices[0].finish_reason
            full_content += content
            if reason == "stop":
                return full_content
            if reason != "length":
                raise RuntimeError(f"Incomplete model response: finish_reason={reason!r}")
            if not content or attempt == 20:
                raise RuntimeError("Model output remains truncated; continuation made no progress or reached its limit")
            messages = initial_messages + [
                {"role": "assistant", "content": full_content},
                {"role": "user", "content": (
                    "Your output was truncated by the output length limit. Continue exactly "
                    "where it ended, including completing any partial line or word. "
                    "Output only the missing suffix, without repeating text, adding a "
                    "preamble, code fences, or an extra separator/newline."
                )},
            ]
        raise RuntimeError("Continuation limit reached")

    def _process_streaming_response(self, response, *, display=True):
        """Collect text and finish reason; parallel workers suppress terminal output."""
        parts = []
        reason = None
        try:
            for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    reason = choice.finish_reason
                content = choice.delta.content
                if content:
                    if display:
                        if not parts:
                            sys.stdout.write(f"{Colors.DIM}  Streaming: {Colors.RESET}")
                        sys.stdout.write(content)
                        sys.stdout.flush()
                    parts.append(content)
        finally:
            response.close()
            if display and parts:
                sys.stdout.write(f"{Colors.RESET}\n")
                sys.stdout.flush()
        return "".join(parts), reason

    def is_toc_page(self, page_image: Image.Image) -> bool:
        """Determine if a page is a table of contents page."""
        base64_image = PDFImageProcessor.convert_to_base64_webp(page_image)
        response = self._send_vision_request([base64_image], TOCDetectionPrompts.IS_TOC_PAGE)
        return "yes" in response.lower()

    def extract_first_arabic_toc_entry(
        self, page_image: Image.Image
    ) -> Optional[Tuple[str, str]]:
        """Extract the first TOC entry that has an Arabic numeral page number (e.g., 1, 5)."""
        import re
        base64_image = PDFImageProcessor.convert_to_base64_webp(page_image)
        response = self._send_vision_request([base64_image], TOCExtractionPrompts.EXTRACT_FIRST_ARABIC_ENTRY)
        cleaned = clean_llm_response(response)

        try:
            if "," not in cleaned:
                return None

            parts = cleaned.split(",", 1)
            page_str = parts[0].strip()
            item_text = parts[1].strip() if len(parts) > 1 else ""

            # Check for 'none' response
            if page_str.lower() == "none" or item_text.lower() == "none":
                return None

            # Use regex to extract just the digits
            digit_match = re.search(r'\d+', page_str)
            if not digit_match:
                return None

            page_num_str = digit_match.group(0)
            page_num = int(page_num_str)

            if page_num < 1:
                return None

            return page_num_str, item_text

        except Exception:
            return None

    def extract_verification_entries(
        self, page_image: Image.Image, first_entry_page: int
    ) -> List[Tuple[int, str]]:
        """Extract 2-3 entries from TOC page that are distant from the first entry for verification."""
        base64_image = PDFImageProcessor.convert_to_base64_webp(page_image)
        prompt = TOCExtractionPrompts.EXTRACT_VERIFICATION_ENTRIES.format(first_entry_page=first_entry_page)
        response = self._send_vision_request([base64_image], prompt, stream=False)
        cleaned = clean_llm_response(response)

        entries = []
        for line in cleaned.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                if ',' in line:
                    page_str, title = line.split(',', 1)
                    page_str = page_str.strip()
                    title = title.strip()
                    if page_str.isdigit() and int(page_str) >= 1:
                        page_num = int(page_str)
                        # Skip if it's the first entry
                        if page_num != first_entry_page:
                            entries.append((page_num, title))
            except Exception:
                continue

        return entries

    def page_contains_content(self, page_image: Image.Image, target_text: str) -> bool:
        """Check if a page contains actual section content (not just header/footer or reference)."""
        base64_image = PDFImageProcessor.convert_to_base64_webp(page_image)
        prompt = ContentVerificationPrompts.PAGE_CONTAINS_CONTENT.format(target_text=target_text)
        response = self._send_vision_request([base64_image], prompt)
        return "yes" in response.lower()

    def verify_offset_match(
        self, page_image: Image.Image, expected_title: str, expected_page: int
    ) -> bool:
        """Verify that this page matches the expected TOC entry for offset validation."""
        base64_image = PDFImageProcessor.convert_to_base64_webp(page_image)
        prompt = ContentVerificationPrompts.VERIFY_OFFSET_MATCH.format(
            expected_title=expected_title,
            expected_page=expected_page
        )
        response = self._send_vision_request([base64_image], prompt, stream=False)
        return "yes" in response.lower()

    def _send_text_request(self, prompt: str, stream: bool = True, timeout: float = 60) -> str:
        """Send a text-only request to the text LLM."""
        return self._complete_text(
            self.text_model, [{"role": "user", "content": prompt}],
            stream=stream, timeout=timeout,
        )

    def _request_refinement(self, messages, tools):
        """Collect complete streamed tool arguments before allowing any edits."""
        parts, calls = [], {}
        reason = None
        last_update = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=self.text_model, messages=messages, tools=tools,
                tool_choice="auto", parallel_tool_calls=False, stream=True,
                timeout=self.refine_timeout,
            )
            try:
                for chunk in response:
                    if time.monotonic() - last_update >= 15:
                        Log.detail("Refinement model is responding; waiting for a complete tool call...")
                        last_update = time.monotonic()
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason is not None:
                        reason = choice.finish_reason
                    delta = choice.delta
                    if delta.content:
                        parts.append(delta.content)
                    for fragment in delta.tool_calls or []:
                        call = calls.setdefault(fragment.index, {
                            "id": "", "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if fragment.id:
                            call["id"] = fragment.id
                        if fragment.function:
                            call["function"]["name"] += fragment.function.name or ""
                            call["function"]["arguments"] += fragment.function.arguments or ""
            finally:
                response.close()
        except APITimeoutError as exc:
            raise RuntimeError(
                f"Refinement request timed out (REFINE_TIMEOUT={self.refine_timeout:g}s). "
                "Increase REFINE_TIMEOUT in model.env and retry with --resume."
            ) from exc
        if any(not call["id"] or not call["function"]["name"] for call in calls.values()):
            raise RuntimeError("Incomplete refinement tool call metadata")
        message = ChatCompletionMessage(
            role="assistant", content="".join(parts) or None,
            tool_calls=[calls[i] for i in sorted(calls)] or None,
        )
        return message, reason

    def refine_bookmarks_with_text_model(self, bookmark_text: str) -> str:
        """Apply validated, exact local edits through a tool conversation."""
        messages = [{"role": "user", "content": BookmarkRefinementPrompts.REFINE_BOOKMARKS.format(
            bookmark_text=bookmark_text
        )}]
        tools = [{"type": "function", "function": {
            "name": "replace_text",
            "description": "Replace one unique exact substring in the CURRENT bookmark text. Use minimal surrounding context to make the match unique.",
            "parameters": {"type": "object", "properties": {
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            }, "required": ["old_text", "new_text"], "additionalProperties": False},
        }}, {"type": "function", "function": {
            "name": "finish_refinement",
            "description": "Finish after all necessary local corrections have been applied, or when no corrections are needed.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        }}]
        current = bookmark_text
        text_only_rounds = 0
        for round_index in range(100):
            Log.detail(f"Refinement round {round_index + 1}: waiting for model (read timeout {self.refine_timeout:g}s)")
            message, finish_reason = self._request_refinement(messages, tools)
            # Thinking-mode providers may reject forced tool choice. Enforce
            # edits through tools locally, even when the API uses auto choice.
            if finish_reason == "stop" and not message.tool_calls:
                text_only_rounds += 1
                if text_only_rounds >= 3:
                    raise RuntimeError("Refinement model repeatedly returned text instead of tool calls")
                messages.append(message.model_dump(exclude_none=True))
                messages.append({"role": "user", "content": (
                    "Your text response has not changed the bookmark document. "
                    "Call replace_text for local edits, or call finish_refinement "
                    "if the current document needs no further changes. Do not output the document."
                )})
                continue
            if finish_reason != "tool_calls" or not message.tool_calls:
                raise RuntimeError(f"Refinement did not return complete tool calls: {finish_reason}")
            text_only_rounds = 0
            messages.append(message.model_dump(exclude_none=True))
            calls = message.tool_calls
            for call in calls:
                try:
                    args = json.loads(call.function.arguments)
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be an object")
                    if call.function.name == "finish_refinement":
                        if len(calls) != 1 or args:
                            raise ValueError("Call finish_refinement alone with no arguments")
                        pattern = (r"BookmarkBegin\nBookmarkTitle: [^\n]+\n"
                                   r"BookmarkLevel: [1-9][0-9]*\n"
                                   r"BookmarkPageNumber: [1-9][0-9]*(?:\n|$)")
                        if current.strip() and re.sub(pattern, "", current.strip()).strip():
                            raise ValueError("Current text is not valid pdftk bookmark blocks; fix formatting before finishing")
                        if bookmark_text.strip() and not current.strip():
                            raise ValueError("Refinement cannot remove all bookmarks")
                        Log.detail(f"Refinement: {len(bookmark_text)} → {len(current)} chars")
                        return current
                    if call.function.name != "replace_text":
                        raise ValueError("Unknown tool")
                    old, new = args.get("old_text"), args.get("new_text")
                    if not isinstance(old, str) or not old or not isinstance(new, str):
                        raise ValueError("old_text must be nonempty and new_text must be a string")
                    if old.strip() == current.strip():
                        raise ValueError("Whole-document replacement is forbidden; edit individual fields or entries")
                    matches = current.count(old)
                    if matches != 1:
                        raise ValueError(f"old_text matches {matches} times; supply exact, unique context")
                    current = current.replace(old, new, 1)
                    result = {"ok": True}
                except (ValueError, TypeError) as exc:
                    result = {"ok": False, "error": str(exc)}
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result, ensure_ascii=False)})
        raise RuntimeError("Refinement tool round limit reached")
