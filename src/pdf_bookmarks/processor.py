"""Main PDF bookmark processor orchestrating the entire workflow."""

from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from pdf_bookmarks.config import Config
from pdf_bookmarks.core import VisionLLMClient, PDFImageProcessor, TOCPageDetector
from pdf_bookmarks.generator import BookmarkGenerator, PDFWriter
from pdf_bookmarks.utils import Log, Colors, clean_llm_response
from pdf_bookmarks.prompts import BookmarkGenerationPrompts
from pdf_bookmarks.progress import ProgressManager, ProgressState
from pdf_bookmarks.signal_handler import get_signal_handler


class PDFBookmarkProcessor:
    """Main class orchestrating the PDF bookmark addition process."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.vision_client = VisionLLMClient(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            vision_model=self.config.vision_model,
            text_model=self.config.text_model,
            refine_timeout=self.config.refine_timeout,
        )
        self.toc_detector = TOCPageDetector(self.vision_client)
        self.bookmark_generator = BookmarkGenerator(self.vision_client)
        self.pdf_writer = PDFWriter(self.config.temp_bookmark_file)

    def process_pdf(self, input_path: str, output_path: str, resume: bool = False, force_restart: bool = False) -> bool:
        """Process a PDF to add bookmarks based on its table of contents.

        Args:
            input_path: Path to the input PDF file
            output_path: Path for output PDF with bookmarks
            resume: Whether to resume from previous progress
            force_restart: Whether to force restart from beginning
        """
        self.progress_manager = ProgressManager(input_path)
        self._current_state = None

        # Setup signal handler for graceful shutdown
        signal_handler = get_signal_handler()
        signal_handler.setup(cleanup_callback=self._save_progress_on_interrupt)

        # Check for existing progress
        if self.progress_manager.exists() and not force_restart:
            state = self.progress_manager.load()
            if self.progress_manager.can_resume(input_path, output_path):
                if state.has_error():
                    # Error state - ask user what to do
                    self.progress_manager.print_resume_info()
                    if not resume:
                        Log.info("Progress file found with error. Use --resume to retry, or delete the .tmp.json file to start fresh.")
                        signal_handler.restore()
                        return False
                    # Clear error and retry
                    state.status = state.get_previous_step()
                    state.error_message = ""
                    self.progress_manager.save(state)
                    signal_handler.restore()
                    return self._resume_processing(input_path, output_path, self.progress_manager)
                elif not resume:
                    self.progress_manager.print_resume_info()
                    Log.info("Use --resume flag to continue from previous progress.")
                    signal_handler.restore()
                    return False
                else:
                    Log.info("Resuming from previous progress...")
                    signal_handler.restore()
                    return self._resume_processing(input_path, output_path, self.progress_manager)

        # Start fresh processing
        result = self._process_fresh(input_path, output_path, self.progress_manager)
        signal_handler.restore()
        return result

    def _save_progress_on_interrupt(self):
        """Save progress when interrupted by signal."""
        if self._current_state is not None:
            try:
                self.progress_manager.save(self._current_state)
                Log.info("Progress saved before exit")
            except Exception as e:
                Log.error(f"Failed to save progress: {e}")

    def _process_fresh(self, input_path: str, output_path: str, progress_manager: ProgressManager) -> bool:
        """Process PDF from scratch with progress tracking."""
        state = None
        try:
            # Initialize progress
            state = ProgressState(
                input_path=input_path,
                output_path=output_path,
                status="scanning_toc"
            )
            self._current_state = state

            # Step 1: Scan for TOC pages
            try:
                toc_pages, content_start_index = self.toc_detector.find_toc_pages(input_path, state, progress_manager)
            except (Exception, KeyboardInterrupt) as e:
                progress_manager.mark_error(state, f"{type(e).__name__}: {e}", "scanning_toc",
                                           page_context=f"Scanning PDF page {state.toc_scan_current_page}")
                Log.info(f"{Colors.YELLOW}Progress saved. Use --resume to continue.{Colors.RESET}")
                raise

            if not toc_pages:
                Log.warn("No TOC pages found in the PDF.")
                progress_manager.delete()
                return False

            state.toc_scan_complete = True
            # toc_pages_count is already updated by detector.find_toc_pages
            state.content_start_index = content_start_index
            state.toc_page_processed = [False] * state.toc_pages_count
            state.status = "calculating_offset"
            progress_manager.save(state)

            Log.info(f"Found {state.toc_pages_count} TOC page(s)")

            # Step 2: Calculate page offset with detailed progress
            try:
                offset_data = self.bookmark_generator.calculate_page_offset_with_progress(
                    input_path, toc_pages, content_start_index, state, progress_manager
                )
                state.offset_calculated = True
                state.page_offset = offset_data["offset"]
                state.first_entry_title = offset_data["first_entry_title"]
                state.first_entry_toc_page = offset_data["first_toc_page"]
                state.first_entry_actual_page = offset_data["actual_page"]
            except (Exception, KeyboardInterrupt) as e:
                progress_manager.mark_error(state, f"{type(e).__name__}: {e}", "calculating_offset",
                                           page_context=f"Searching page {state.offset_search_current_page}")
                Log.info(f"{Colors.YELLOW}Progress saved. Use --resume to continue.{Colors.RESET}")
                raise

            # Step 3: Verify offset
            state.status = "verifying_offset"
            progress_manager.save(state)

            try:
                self._verify_offset_with_progress(toc_pages, state, progress_manager)
            except (Exception, KeyboardInterrupt) as e:
                progress_manager.mark_error(state, f"{type(e).__name__}: {e}", "verifying_offset",
                                           page_context=f"Verification {state.verification_current}/2")
                Log.info(f"{Colors.YELLOW}Progress saved. Use --resume to continue.{Colors.RESET}")
                raise

            state.verification_passed = True

            # Step 4: Generate bookmarks with progress tracking
            state.status = "generating_bookmarks"
            state.current_toc_page_index = 0
            progress_manager.save(state)

            Log.step("Generating bookmark structure...")
            try:
                bookmark_text = self._generate_bookmarks_with_progress(
                    toc_pages, state, progress_manager
                )
            except (Exception, KeyboardInterrupt) as e:
                progress_manager.mark_error(state, f"{type(e).__name__}: {e}", "generating_bookmarks",
                                           page_context=f"Processing TOC page {state.current_toc_page_index + 1}/{len(toc_pages)}")
                Log.info(f"{Colors.YELLOW}Progress saved. Use --resume to continue.{Colors.RESET}")
                raise

            if not bookmark_text.strip():
                Log.warn("No valid (Arabic-numbered) bookmark entries found. Exiting.")
                progress_manager.delete()
                return False

            state.accumulated_bookmarks = bookmark_text
            state.status = "refining_bookmarks"
            progress_manager.save(state)

            # Step 5: Refine bookmarks
            Log.step("Refining bookmarks with text model...")
            try:
                bookmark_text = self.vision_client.refine_bookmarks_with_text_model(bookmark_text)
            except (Exception, KeyboardInterrupt) as e:
                progress_manager.mark_error(state, f"{type(e).__name__}: {e}", "refining_bookmarks")
                Log.info(f"{Colors.YELLOW}Progress saved. Use --resume to continue.{Colors.RESET}")
                raise

            state.refined_bookmarks = bookmark_text
            state.status = "applying_bookmarks"
            progress_manager.save(state)

            # Step 6: Apply offset and generate PDF
            bookmark_text = self.bookmark_generator.apply_page_offset(
                bookmark_text, state.page_offset
            )

            try:
                self.pdf_writer.add_bookmarks_to_pdf(
                    bookmark_text, input_path, output_path
                )
            except (Exception, KeyboardInterrupt) as e:
                progress_manager.mark_error(state, f"{type(e).__name__}: {e}", "applying_bookmarks")
                Log.info(f"{Colors.YELLOW}Progress saved. Use --resume to continue.{Colors.RESET}")
                raise

            # Complete - delete progress file
            state.status = "completed"
            state.error_message = ""
            progress_manager.save(state)
            progress_manager.delete()

            Log.separator()
            Log.success("PDF bookmark processing completed successfully!")
            Log.separator()
            return True

        except KeyboardInterrupt:
            Log.warn(f"\n{Colors.YELLOW}Processing interrupted by user.")
            Log.info(f"Progress has been saved. Use {Colors.BOLD}--resume{Colors.RESET}{Colors.YELLOW} to continue.{Colors.RESET}")
            return False
        except Exception as e:
            Log.error(f"Error processing PDF: {e}")
            # Error is already marked in progress manager
            return False

    def _verify_offset_with_progress(
        self, toc_pages, state: ProgressState, progress_manager: ProgressManager
    ) -> None:
        """Verify offset with progress tracking. If verification fails, try offset+1."""
        verification_entries = self.vision_client.extract_verification_entries(
            toc_pages[0], state.first_entry_toc_page
        )

        if not verification_entries or len(verification_entries) < 2:
            Log.warn("Could not extract enough entries for verification. Using initial offset.")
            return

        # Try verification with current offset first
        verification_passed = self._verify_offset_with_specific_value(
            toc_pages, state, progress_manager, verification_entries, state.page_offset, "current"
        )

        # If verification failed, try offset+1 and offset-1 (if -1 doesn't point to TOC pages)
        if verification_passed < 1:
            from pypdf import PdfReader
            pdf_reader = PdfReader(state.input_path)
            total_pages = len(pdf_reader.pages)

            # Calculate TOC page range (0-based indices)
            toc_range_start = state.toc_start_index if state.toc_start_index >= 0 else 0
            toc_range_end = state.content_start_index - 1 if state.content_start_index > 0 else 0

            # Helper to check if an offset would point to TOC pages
            def offset_points_to_toc(offset: int) -> bool:
                """Check if any verification entry with this offset would point to a TOC page."""
                for toc_page_num, entry_title in verification_entries[:2]:
                    expected_page = toc_page_num + offset
                    # Check if page is valid and within TOC range
                    if 1 <= expected_page <= total_pages:
                        # Convert to 0-based index
                        page_index = expected_page - 1
                        if toc_range_start <= page_index <= toc_range_end:
                            Log.detail(f"Offset {offset} would point '{entry_title}' to TOC page {expected_page} (within TOC range {toc_range_start+1}-{toc_range_end+1})")
                            return True
                return False

            # Try offset+1
            offset_plus_one = state.page_offset + 1
            Log.warn(f"Offset verification failed. Trying offset+1 ({offset_plus_one})...")
            verification_passed_plus_one = self._verify_offset_with_specific_value(
                toc_pages, state, progress_manager, verification_entries, offset_plus_one, "+1"
            )

            # Try offset-1 if it doesn't point to TOC pages
            offset_minus_one = state.page_offset - 1
            verification_passed_minus_one = 0
            should_try_minus_one = offset_minus_one >= 0 and not offset_points_to_toc(offset_minus_one)

            if should_try_minus_one:
                Log.warn(f"Also trying offset-1 ({offset_minus_one})...")
                verification_passed_minus_one = self._verify_offset_with_specific_value(
                    toc_pages, state, progress_manager, verification_entries, offset_minus_one, "-1"
                )
            else:
                Log.detail(f"Skipping offset-1 ({offset_minus_one}) because it would point to TOC pages")

            # Determine which offset (if any) passed verification
            best_offset = None
            best_passed = 0

            if verification_passed_plus_one >= 1:
                best_offset = offset_plus_one
                best_passed = verification_passed_plus_one
                Log.success(f"Offset+1 passed verification ({verification_passed_plus_one}/2)")

            if verification_passed_minus_one >= 1 and verification_passed_minus_one > best_passed:
                best_offset = offset_minus_one
                best_passed = verification_passed_minus_one
                Log.success(f"Offset-1 passed verification ({verification_passed_minus_one}/2)")

            if best_offset is not None:
                # Update state with new offset
                state.page_offset = best_offset
                state.verification_passed = True
                Log.success(f"Offset adjusted to {state.page_offset} ({'+1' if best_offset == offset_plus_one else '-1'})")
                progress_manager.save(state)
                verification_passed = best_passed
            else:
                raise RuntimeError(
                    f"Offset verification failed with all tried offsets (+1: {verification_passed_plus_one}/2, "
                    f"-1: {verification_passed_minus_one}/2). "
                    "The calculated offset may be incorrect. Please check the PDF structure."
                )
        else:
            state.verification_passed = True
            progress_manager.save(state)

        Log.info(f"Offset verification passed ({verification_passed}/2 confirmations)")

    def _verify_offset_with_specific_value(
        self, toc_pages, state: ProgressState, progress_manager: ProgressManager,
        verification_entries, offset_to_test: int, label: str = "test"
    ) -> int:
        """Verify offset with a specific offset value. Returns number of passed verifications."""
        Log.detail(f"Testing offset {label} ({offset_to_test})...")

        # Clear previous verification entries for this test
        if label == "current":
            state.verification_entries = []

        verification_passed = 0

        for i, (toc_page_num, entry_title) in enumerate(verification_entries[:2], 1):
            if label != "current":
                state.verification_current = i
            expected_page = toc_page_num + offset_to_test
            Log.detail(f"Verification {i}: Checking '{entry_title}' at expected page {expected_page} (offset {offset_to_test})")

            from pypdf import PdfReader
            if expected_page < 1 or expected_page > len(PdfReader(state.input_path).pages):
                Log.warn(f"Verification {i}: Expected page {expected_page} is out of range. Skipping.")
                if label == "current":
                    state.verification_entries.append({
                        "index": i,
                        "title": entry_title,
                        "toc_page": toc_page_num,
                        "page": expected_page,
                        "passed": False,
                        "skipped": True
                    })
                    progress_manager.save(state)
                continue

            page_image = PDFImageProcessor.extract_page_as_image(state.input_path, expected_page - 1)
            verified = self.vision_client.verify_offset_match(page_image, entry_title, expected_page)

            if label == "current":
                state.verification_entries.append({
                    "index": i,
                    "title": entry_title,
                    "toc_page": toc_page_num,
                    "page": expected_page,
                    "passed": verified
                })
                progress_manager.save(state)

            if verified:
                Log.success(f"Verification {i}: Confirmed (offset {offset_to_test})")
                verification_passed += 1
            else:
                Log.warn(f"Verification {i}: Failed - content not found at expected page {expected_page} (offset {offset_to_test})")

        return verification_passed

    def _resume_processing(self, input_path: str, output_path: str, progress_manager: ProgressManager) -> bool:
        """Resume processing from saved progress."""
        state = progress_manager.load()
        self._current_state = state

        try:
            Log.info(f"Resuming from status: {state.get_progress_summary()}")

            # Initialize variables that will be populated during resume
            toc_pages = None
            content_start_index = 0

            if state.status == "scanning_toc":
                # Continue scanning for TOC pages from where we left off
                state.error_message = ""
                state.error_step = ""
                toc_pages, content_start_index = self.toc_detector.find_toc_pages(input_path, state, progress_manager)

                if not toc_pages:
                    # Check if we hit content start during resume (means TOC scan completed previously)
                    if content_start_index > 0:
                        # We've found content start page. Check if we have saved TOC page indices
                        if state.toc_start_index >= 0 and state.toc_pages_count > 0:
                            # Check if we have scanned all pages between toc_start_index and content_start_index
                            # toc_scan_current_page is 1-based, content_start_index is 0-based
                            if state.toc_scan_current_page - 1 < content_start_index:
                                # We haven't scanned all pages, need to rescan to ensure completeness
                                Log.info(f"Some pages between {state.toc_start_index + 1} and {content_start_index + 1} were not scanned. Rescanning...")
                                # Reset state to scan from toc_start_index
                                state.toc_scan_current_page = state.toc_start_index
                                progress_manager.save(state)
                                toc_pages, content_start_index = self.toc_detector.find_toc_pages(input_path, state, progress_manager)
                                if not toc_pages:
                                    Log.warn("No TOC pages found after rescan.")
                                    progress_manager.delete()
                                    return False
                            else:
                                # We've scanned all pages, extract directly
                                Log.info(f"Using saved TOC info: {state.toc_pages_count} pages starting from PDF page {state.toc_start_index + 1}")
                                toc_pages = self.toc_detector.extract_toc_pages_direct(
                                    input_path, state.toc_start_index, state.toc_pages_count
                                )
                                content_start_index = state.content_start_index
                                Log.info(f"Extracted {len(toc_pages)} TOC page(s)")
                        else:
                            # We don't have TOC info saved, need to scan from beginning
                            Log.info(f"TOC scan completed. Finding all TOC pages up to page {content_start_index + 1}...")
                            # Reset state to scan from beginning
                            state.toc_scan_current_page = 0
                            progress_manager.save(state)
                            toc_pages, content_start_index = self.toc_detector.find_toc_pages(input_path, state, progress_manager)
                            if not toc_pages:
                                Log.warn("No TOC pages found in the PDF.")
                                progress_manager.delete()
                                return False
                    else:
                        Log.warn("No TOC pages found in the PDF.")
                        progress_manager.delete()
                        return False

                state.toc_scan_complete = True
                # toc_pages_count is already updated by detector.find_toc_pages
                state.content_start_index = content_start_index
                # Initialize toc_page_processed if not already set
                if not state.toc_page_processed or len(state.toc_page_processed) != state.toc_pages_count:
                    state.toc_page_processed = [False] * state.toc_pages_count
                state.status = "calculating_offset"
                progress_manager.save(state)
                Log.info(f"Found {state.toc_pages_count} TOC page(s)")

            if state.status == "calculating_offset":
                # Get TOC pages - either from saved info or by scanning
                if toc_pages is None:
                    if state.toc_scan_complete and state.toc_pages_count > 0:
                        # TOC scan was complete, extract pages directly
                        toc_start_index = state.toc_start_index if state.toc_start_index >= 0 else state.content_start_index - state.toc_pages_count
                        Log.step(f"Using saved TOC info: {state.toc_pages_count} pages starting from PDF page {toc_start_index + 1}")
                        toc_pages = self.toc_detector.extract_toc_pages_direct(
                            input_path, toc_start_index, state.toc_pages_count
                        )
                        content_start_index = state.content_start_index
                    else:
                        # Need to scan for TOC pages
                        state.toc_scan_current_page = 0
                        toc_pages, content_start_index = self.toc_detector.find_toc_pages(input_path, state, progress_manager)
                        if not toc_pages:
                            Log.warn("No TOC pages found in the PDF.")
                            progress_manager.delete()
                            return False
                        # Save the TOC scan info
                        state.toc_scan_complete = True
                        # toc_pages_count and content_start_index are already updated by detector.find_toc_pages
                        # Initialize toc_page_processed if not already set
                        if not state.toc_page_processed or len(state.toc_page_processed) != state.toc_pages_count:
                            state.toc_page_processed = [False] * state.toc_pages_count
                        progress_manager.save(state)

                try:
                    offset_data = self.bookmark_generator.calculate_page_offset_with_progress(
                        input_path, toc_pages, content_start_index, state, progress_manager
                    )
                    state.offset_calculated = True
                    state.page_offset = offset_data["offset"]
                    state.first_entry_title = offset_data["first_entry_title"]
                    state.first_entry_toc_page = offset_data["first_toc_page"]
                    state.first_entry_actual_page = offset_data["actual_page"]
                except (Exception, KeyboardInterrupt) as e:
                    progress_manager.mark_error(state, f"{type(e).__name__}: {e}", "calculating_offset",
                                               page_context=f"Searching page {state.offset_search_current_page}")
                    Log.info(f"{Colors.YELLOW}Progress saved. Use --resume to continue.{Colors.RESET}")
                    raise

                # Proceed to verification
                self._verify_offset_with_progress(toc_pages, state, progress_manager)
                state.verification_passed = True
                state.status = "generating_bookmarks"
                progress_manager.save(state)

            if state.status == "verifying_offset":
                # Get TOC pages - either from saved info or by scanning
                if toc_pages is None:
                    if state.toc_scan_complete and state.toc_pages_count > 0:
                        toc_start_index = state.toc_start_index if state.toc_start_index >= 0 else state.content_start_index - state.toc_pages_count
                        toc_pages = self.toc_detector.extract_toc_pages_direct(
                            input_path, toc_start_index, state.toc_pages_count
                        )
                        content_start_index = state.content_start_index
                    else:
                        state.toc_scan_current_page = 0
                        toc_pages, content_start_index = self.toc_detector.find_toc_pages(input_path, state, progress_manager)
                self._verify_offset_with_progress(toc_pages, state, progress_manager)
                state.verification_passed = True
                state.status = "generating_bookmarks"
                progress_manager.save(state)

            if state.status == "generating_bookmarks":
                # Get TOC pages - either from saved info or by scanning
                if toc_pages is None:
                    if state.toc_scan_complete and state.toc_pages_count > 0:
                        toc_start_index = state.toc_start_index if state.toc_start_index >= 0 else state.content_start_index - state.toc_pages_count
                        toc_pages = self.toc_detector.extract_toc_pages_direct(
                            input_path, toc_start_index, state.toc_pages_count
                        )
                        content_start_index = state.content_start_index
                    else:
                        state.toc_scan_current_page = 0
                        toc_pages, content_start_index = self.toc_detector.find_toc_pages(input_path, state, progress_manager)

                # Check if we need to reset and generate fresh bookmarks
                # (e.g., coming from verification step, not resuming a partial generation)
                needs_fresh_generation = (
                    not state.toc_page_results and (
                        not state.accumulated_bookmarks or
                        "BookmarkBegin" not in state.accumulated_bookmarks
                    )
                )

                if needs_fresh_generation:
                    state.accumulated_bookmarks = ""
                    state.last_entry = ""
                    state.current_toc_page_index = 0
                    state.toc_page_processed = [False] * len(toc_pages)
                    progress_manager.save(state)

                Log.step("Generating bookmark structure...")
                try:
                    bookmark_text = self._generate_bookmarks_with_progress(
                        toc_pages, state, progress_manager
                    )
                except (Exception, KeyboardInterrupt) as e:
                    progress_manager.mark_error(state, f"{type(e).__name__}: {e}", "generating_bookmarks",
                                               page_context=f"Processing TOC page {state.current_toc_page_index + 1}/{len(toc_pages)}")
                    Log.info(f"{Colors.YELLOW}Progress saved. Use --resume to continue.{Colors.RESET}")
                    raise

                state.accumulated_bookmarks = bookmark_text
                state.status = "refining_bookmarks"
                state.error_message = ""
                progress_manager.save(state)

            # Continue from refinement or apply
            if state.status in ("refining_bookmarks", "applying_bookmarks") or not state.refined_bookmarks:
                Log.step("Refining bookmarks with text model...")
                try:
                    bookmark_text = self.vision_client.refine_bookmarks_with_text_model(
                        state.accumulated_bookmarks
                    )
                except (Exception, KeyboardInterrupt) as e:
                    progress_manager.mark_error(state, f"{type(e).__name__}: {e}", "refining_bookmarks")
                    Log.info(f"{Colors.YELLOW}Progress saved. Use --resume to continue.{Colors.RESET}")
                    raise

                state.refined_bookmarks = bookmark_text
                state.status = "applying_bookmarks"
                state.error_message = ""
                progress_manager.save(state)

            # Apply offset and generate PDF
            bookmark_text = self.bookmark_generator.apply_page_offset(
                state.refined_bookmarks, state.page_offset
            )

            try:
                self.pdf_writer.add_bookmarks_to_pdf(
                    bookmark_text, input_path, output_path
                )
            except (Exception, KeyboardInterrupt) as e:
                progress_manager.mark_error(state, f"{type(e).__name__}: {e}", "applying_bookmarks")
                Log.info(f"{Colors.YELLOW}Progress saved. Use --resume to continue.{Colors.RESET}")
                raise

            state.status = "completed"
            state.error_message = ""
            progress_manager.save(state)
            progress_manager.delete()

            Log.separator()
            Log.success("PDF bookmark processing completed successfully!")
            Log.separator()
            return True

        except KeyboardInterrupt:
            Log.warn(f"\n{Colors.YELLOW}Processing interrupted by user.")
            Log.info(f"Progress has been saved. Use {Colors.BOLD}--resume{Colors.RESET}{Colors.YELLOW} to continue.{Colors.RESET}")
            return False
        except Exception as e:
            Log.error(f"Error processing PDF: {e}")
            # Error is already marked in progress manager
            return False

    def _generate_bookmarks_with_progress(
        self, toc_pages, state: ProgressState, progress_manager: ProgressManager
    ) -> str:
        """Generate bookmarks with progress tracking for resumability."""
        count = len(toc_pages)
        # Legacy checkpoints only contain a combined prefix. Re-extract once so
        # every saved result has a reliable page identity.
        if not state.toc_page_results:
            state.toc_page_processed = [False] * count
            state.accumulated_bookmarks = ""
        state.toc_page_processed = [str(i) in state.toc_page_results for i in range(count)]

        def save_results():
            state.accumulated_bookmarks = "\n".join(
                state.toc_page_results[str(i)] for i in range(count)
                if state.toc_page_results.get(str(i))
            )
            state.total_bookmarks_generated = state.accumulated_bookmarks.count("BookmarkBegin")
            state.current_toc_page_index = next(
                (i for i in range(count) if not state.toc_page_processed[i]), count
            )
            progress_manager.save(state)

        def extract_page(i):
            base64_image = PDFImageProcessor.convert_to_base64_webp(toc_pages[i])
            response = self.vision_client._send_vision_request(
                [base64_image], BookmarkGenerationPrompts.FIRST_PAGE_PROMPT, display=False,
            )
            return clean_llm_response(response).strip()

        save_results()
        pending = [i for i in range(count) if not state.toc_page_processed[i]]
        if not pending:
            return state.accumulated_bookmarks.strip()
        workers = max(1, min(self.config.toc_workers, len(pending)))
        Log.detail(f"Extracting {len(pending)} TOC pages independently with {workers} workers")
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {executor.submit(extract_page, i): i for i in pending}
        failures = []
        try:
            for future in as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failures.append((i, exc))
                    Log.warn(f"TOC page {i + 1} failed: {exc}")
                    continue
                state.toc_page_results[str(i)] = result
                state.toc_page_processed[i] = True
                save_results()
                Log.detail(f"TOC page {i + 1}/{count} saved ({sum(state.toc_page_processed)}/{count} complete)")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if failures:
            i, exc = failures[0]
            state.current_toc_page_index = i
            raise RuntimeError(f"TOC page {i + 1} failed: {exc}") from exc
        return state.accumulated_bookmarks.strip()
