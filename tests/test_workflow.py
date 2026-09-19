import copy
import json
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from pdf_bookmarks.core.llm import VisionLLMClient
from pdf_bookmarks.processor import PDFBookmarkProcessor
from pdf_bookmarks.progress.state import ProgressState


def bookmark(title):
    return f'BookmarkBegin\nBookmarkTitle: {title}\nBookmarkLevel: 1\nBookmarkPageNumber: 1'


def stream(text, reason):
    response = Mock()
    response.__iter__ = Mock(return_value=iter([
        NS(choices=[NS(delta=NS(content=text), finish_reason=None)]),
        NS(choices=[NS(delta=NS(content=None), finish_reason=reason)]),
    ]))
    return response


def refinement_stream(choice):
    response = Mock(choices=[choice])
    def chunks():
        yield NS(choices=[])  # usage/heartbeat chunk
        content = choice.message.model_dump.return_value.get('content')
        yield NS(choices=[NS(delta=NS(content=content, tool_calls=None), finish_reason=None)])
        for index, call in enumerate(choice.message.tool_calls or []):
            yield NS(choices=[NS(delta=NS(content=None, tool_calls=[NS(
                index=index, id=call.id, function=NS(name=call.function.name, arguments='')
            )]), finish_reason=None)])
            for char in call.function.arguments:
                yield NS(choices=[NS(delta=NS(content=None, tool_calls=[NS(
                    index=index, id=None, function=NS(name=None, arguments=char)
                )]), finish_reason=None)])
        yield NS(choices=[NS(delta=NS(content=None, tool_calls=None), finish_reason=choice.finish_reason)])
    response.__iter__ = Mock(side_effect=chunks)
    return response


def tool_response(name, args):
    call = NS(id='call', function=NS(name=name, arguments=json.dumps(args)))
    message = Mock(tool_calls=[call])
    message.model_dump.return_value = {'role': 'assistant', 'tool_calls': [
        {'id': call.id, 'type': 'function', 'function': vars(call.function)}]}
    return refinement_stream(NS(finish_reason='tool_calls', message=message))


class WorkflowTests(unittest.TestCase):
    def client(self):
        client = VisionLLMClient.__new__(VisionLLMClient)
        client.client = Mock()
        client.text_model = client.vision_model = 'test'
        client.refine_timeout = 600
        return client

    def test_continuation_preserves_partial_line_and_context(self):
        client = self.client()
        responses = [stream('BookmarkPageNum', 'length'), stream('ber: ', 'length'), stream('12', 'stop')]
        client.client.chat.completions.create.side_effect = responses
        result = client._send_vision_request(['image'], 'extract', display=False)
        self.assertEqual(result, 'BookmarkPageNumber: 12')
        requests = client.client.chat.completions.create.call_args_list
        self.assertEqual(requests[2].kwargs['messages'][1]['content'], 'BookmarkPageNumber: ')
        self.assertEqual(requests[2].kwargs['messages'][0], requests[0].kwargs['messages'][0])
        for response in responses:
            response.close.assert_called_once()

    def test_unfinished_stream_is_not_success(self):
        client = self.client()
        client.client.chat.completions.create.return_value = stream('partial', None)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            client._send_vision_request([], 'extract', display=False)

    def test_empty_truncated_response_fails(self):
        client = self.client()
        client.client.chat.completions.create.return_value = stream('', 'length')
        with self.assertRaisesRegex(RuntimeError, 'no progress'):
            client._send_vision_request([], 'extract', display=False)

    def test_refinement_changes_only_matching_text(self):
        client = self.client()
        original = bookmark('A') + '\n' + bookmark('B')
        client.client.chat.completions.create.side_effect = [
            tool_response('replace_text', {'old_text': 'BookmarkTitle: B', 'new_text': 'BookmarkTitle: C'}),
            tool_response('finish_refinement', {}),
        ]
        self.assertEqual(client.refine_bookmarks_with_text_model(original), original.replace('Title: B', 'Title: C'))

    def test_ambiguous_edit_and_whole_document_rejected(self):
        client = self.client()
        original = bookmark('A') + '\n' + bookmark('B')
        results = []
        responses = iter([
            tool_response('replace_text', {'old_text': 'BookmarkLevel: 1', 'new_text': 'BookmarkLevel: 2'}),
            tool_response('replace_text', {'old_text': original, 'new_text': bookmark('C')}),
            tool_response('finish_refinement', {}),
        ])
        def create(**kwargs):
            results.append(copy.deepcopy(kwargs['messages']))
            return next(responses)
        client.client.chat.completions.create.side_effect = create
        self.assertEqual(client.refine_bookmarks_with_text_model(original), original)
        self.assertIn('2 times', results[1][-1]['content'])
        self.assertIn('Whole-document', results[2][-1]['content'])

    def test_truncated_tool_is_never_applied(self):
        client = self.client()
        response = tool_response('replace_text', {'old_text': 'A', 'new_text': 'B'})
        response.choices[0].finish_reason = 'length'
        client.client.chat.completions.create.return_value = response
        with self.assertRaisesRegex(RuntimeError, 'complete tool calls'):
            client.refine_bookmarks_with_text_model(bookmark('A'))

    def test_auto_tool_choice_reprompts_without_applying_text(self):
        client = self.client()
        message = Mock(tool_calls=None)
        message.model_dump.return_value = {'role': 'assistant', 'content': bookmark('WRONG')}
        client.client.chat.completions.create.side_effect = [
            refinement_stream(NS(finish_reason='stop', message=message)),
            tool_response('finish_refinement', {}),
        ]
        self.assertEqual(client.refine_bookmarks_with_text_model(bookmark('A')), bookmark('A'))
        for call in client.client.chat.completions.create.call_args_list:
            self.assertEqual(call.kwargs['tool_choice'], 'auto')
            self.assertTrue(call.kwargs['stream'])
            self.assertEqual(call.kwargs['timeout'], 600)

    def test_text_only_refinement_has_bounded_retries(self):
        client = self.client()
        message = Mock(tool_calls=None)
        message.model_dump.return_value = {'role': 'assistant', 'content': 'Done'}
        client.client.chat.completions.create.return_value = refinement_stream(
            NS(finish_reason='stop', message=message))
        with self.assertRaisesRegex(RuntimeError, 'repeatedly returned text'):
            client.refine_bookmarks_with_text_model(bookmark('A'))
        self.assertEqual(client.client.chat.completions.create.call_count, 3)

    def test_refinement_resume_skips_extraction(self):
        state = ProgressState('in', 'out', 'error', error_step='refining_bookmarks',
                              accumulated_bookmarks=bookmark('A'))
        state.status = state.get_previous_step()
        self.assertEqual(state.status, 'refining_bookmarks')
        processor = PDFBookmarkProcessor.__new__(PDFBookmarkProcessor)
        processor.vision_client = Mock()
        processor.vision_client.refine_bookmarks_with_text_model.return_value = bookmark('A')
        processor.bookmark_generator = Mock()
        processor.pdf_writer = Mock()
        processor.toc_detector = Mock()
        manager = Mock()
        manager.load.return_value = state
        self.assertTrue(processor._resume_processing('in', 'out', manager))
        processor.vision_client.refine_bookmarks_with_text_model.assert_called_once_with(bookmark('A'))
        processor.toc_detector.extract_toc_pages_direct.assert_not_called()
        processor.toc_detector.find_toc_pages.assert_not_called()

    def test_refinement_stream_timeout_closes_response(self):
        from openai import APITimeoutError
        import httpx
        client = self.client()
        client.refine_timeout = 900
        response = Mock()
        def interrupted():
            yield NS(choices=[])
            raise APITimeoutError(request=httpx.Request('POST', 'https://example.invalid'))
        response.__iter__ = Mock(side_effect=interrupted)
        client.client.chat.completions.create.return_value = response
        with self.assertRaisesRegex(RuntimeError, 'REFINE_TIMEOUT=900s'):
            client.refine_bookmarks_with_text_model(bookmark('A'))
        response.close.assert_called_once()

    def test_parallel_failure_and_resume_preserve_page_order(self):
        processor = PDFBookmarkProcessor.__new__(PDFBookmarkProcessor)
        processor.config = NS(toc_workers=3)
        processor.vision_client = Mock()
        state = ProgressState('in', 'out', 'generating_bookmarks', toc_pages_count=3)
        barrier = threading.Barrier(3)
        snapshots = []
        manager = Mock()
        manager.save.side_effect = lambda state: snapshots.append(copy.deepcopy(state.to_dict()))
        def extract(images, prompt, **kwargs):
            barrier.wait(timeout=3)  # All requests must be in flight at once.
            if images[0] == 1:
                raise RuntimeError('retry me')
            return bookmark(str(images[0]))
        processor.vision_client._send_vision_request.side_effect = extract
        with patch('pdf_bookmarks.processor.PDFImageProcessor.convert_to_base64_webp', side_effect=lambda p: p):
            with self.assertRaisesRegex(RuntimeError, 'TOC page 2'):
                processor._generate_bookmarks_with_progress([0, 1, 2], state, manager)
            self.assertEqual(state.toc_page_processed, [True, False, True])
            self.assertEqual(len(set(c.args[1] for c in processor.vision_client._send_vision_request.call_args_list)), 1)
            state = ProgressState.from_dict(snapshots[-1])
            processor.vision_client._send_vision_request.reset_mock(side_effect=True)
            processor.vision_client._send_vision_request.return_value = bookmark('1')
            result = processor._generate_bookmarks_with_progress([0, 1, 2], state, manager)
            processor.vision_client._send_vision_request.assert_called_once()
            self.assertEqual(result, '\n'.join(bookmark(str(i)) for i in range(3)))


if __name__ == '__main__':
    unittest.main()
