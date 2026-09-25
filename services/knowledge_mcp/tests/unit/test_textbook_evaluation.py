import math
from unittest.mock import AsyncMock, MagicMock, patch

from scripts.evaluate_textbook_ragas import finite, mean_valid
from src.core.types import RetrievalResult
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.langfuse_exporter import export_evaluation


def test_custom_accepts_real_retrieval_results():
    chunks = [RetrievalResult(chunk_id='a', score=0.9, text='a', metadata={}),
              RetrievalResult(chunk_id='b', score=0.8, text='b', metadata={})]
    assert CustomEvaluator().evaluate('q', chunks, ground_truth=['b']) == {'hit_rate': 1.0, 'mrr': 0.5}


def test_undefined_is_not_zero_and_denominator_is_explicit():
    assert finite(float('nan')) is None
    assert finite(None) is None
    assert finite(0) == 0
    assert mean_valid([1.0, None, 0.0]) == {'mean': 0.5, 'valid_n': 2, 'undefined_n': 1}


def test_eval_runner_excludes_nan_and_emits_valid_json():
    import json
    from src.observability.evaluation.eval_runner import EvalRunner, EvalReport, QueryResult
    rows = [QueryResult(query='a',metrics={'faithfulness':float('nan')}),
            QueryResult(query='b',metrics={'faithfulness':0.8})]
    average = EvalRunner._aggregate_metrics(rows)
    assert average['faithfulness'] == 0.8
    report = EvalReport(query_results=rows,aggregate_metrics=average).to_dict()
    assert report['metric_valid_counts']['faithfulness'] == 1
    assert report['query_results'][0]['metrics']['faithfulness'] is None
    json.dumps(report,allow_nan=False)


def test_ragas_adapter_does_not_turn_missing_score_into_zero():
    from src.observability.evaluation.ragas_evaluator import RagasEvaluator
    evaluator = RagasEvaluator(metrics=['faithfulness'])
    evaluator._build_wrappers = MagicMock(return_value=(MagicMock(), MagicMock()))
    with patch('ragas.metrics.collections.Faithfulness') as metric:
        metric.return_value.ascore = AsyncMock(return_value=MagicMock(value=None))
        assert math.isnan(evaluator.evaluate('q',['context'],'answer')['faithfulness'])


def test_ragas_metrics_share_one_running_loop():
    import asyncio
    from src.observability.evaluation.ragas_evaluator import RagasEvaluator
    loops=[]
    async def scoring(**kwargs):
        loops.append(asyncio.get_running_loop())
        return MagicMock(value=1.0)
    evaluator=RagasEvaluator()
    evaluator._build_wrappers=MagicMock(return_value=(MagicMock(),MagicMock()))
    with patch('ragas.metrics.collections.Faithfulness') as f, patch('ragas.metrics.collections.AnswerRelevancy') as a, patch('ragas.metrics.collections.ContextPrecisionWithoutReference') as c:
        for metric in [f,a,c]: metric.return_value.ascore=AsyncMock(side_effect=scoring)
        evaluator.evaluate('q',['context'],'answer')
    assert len(loops)==3
    assert all(loop is loops[0] for loop in loops)


def test_langfuse_is_disabled_without_import_or_network():
    client = MagicMock()
    assert export_evaluation([], [], 'run', client=client) == {'status':'disabled','exported':0}
    client.auth_check.assert_not_called()


def test_langfuse_missing_configuration_is_explicit():
    with patch.dict('os.environ', {}, clear=True):
        result = export_evaluation([], [], 'run', enabled=True)
    assert result['status'] == 'not_configured'
    assert 'LANGFUSE_SECRET_KEY' in result['missing']


def test_langfuse_skips_undefined_scores_and_marks_replay():
    client = MagicMock()
    client.auth_check.return_value = True
    span = client.start_as_current_observation.return_value.__enter__.return_value
    span.trace_id = 'trace'
    sample = dict(id='q',variant='baseline',split='dev',user_input='q',response='a',chunks=[],ir={})
    result = dict(id='q',variant='baseline',status={'faithfulness':'undefined','answer_relevancy':'ok'},
                  metrics={'faithfulness':None,'answer_relevancy':0.8})
    result = export_evaluation([sample], [result], 'run', enabled=True, client=client)
    assert result['exported'] == 1
    assert client.start_as_current_observation.call_args.kwargs['metadata']['replayed'] is True
    assert span.score.call_count == 1
    client.flush.assert_called_once()
