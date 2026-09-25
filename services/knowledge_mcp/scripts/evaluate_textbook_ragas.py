"""Resume-safe Ragas evaluation of immutable, paired textbook pipeline outputs."""
import argparse
import asyncio
import contextvars
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['RAGAS_DO_NOT_TRACK'] = 'true'
from src.core.settings import load_settings
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.evaluation.ragas_evaluator import RagasEvaluator

CURRENT = contextvars.ContextVar('evaluation', default={})
METRICS = ['faithfulness', 'answer_relevancy', 'context_precision_without_reference']

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def hashed(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None

def mean_valid(values):
    valid = [x for x in values if x is not None and math.isfinite(x)]
    return dict(mean=statistics.mean(valid) if valid else None, valid_n=len(valid), undefined_n=len(values)-len(valid))

class Recorder:
    """Record JSON payloads only. Never persist headers, credentials, or environment."""
    def __init__(self, output):
        self.output = output
        self.next_request = 0.0
        self.lock = asyncio.Lock()

    async def request(self, request):
        # Bound request rate across workers. SDK handles transient 429 retries.
        if request.url.path.endswith('/chat/completions'):
            async with self.lock:
                await asyncio.sleep(max(0, self.next_request-time.monotonic()))
                self.next_request = time.monotonic()+1.5
        request.extensions['audit_start'] = time.monotonic()

    async def response(self, response):
        await response.aread()
        request = response.request
        try:
            data = response.json()
            body = json.loads(request.content)
        except (ValueError, UnicodeDecodeError):
            return
        # Error messages may contain service/account details: record only status/code.
        record = dict(id=uuid.uuid4().hex, **CURRENT.get(), path=request.url.path,
                      status=response.status_code, seconds=time.monotonic()-request.extensions['audit_start'],
                      request=body, usage=data.get('usage'), response_model=data.get('model'))
        if response.is_success:
            record['response'] = {k:v for k,v in data.items() if k != 'data'}  # omit embedding vectors
        else:
            record['error_code'] = data.get('error',{}).get('code')
        with (self.output/'api_calls.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')

def build_samples(source):
    old = read(ROOT/'data/textbook_ch1_3/evaluation/all_results.json')
    split = read(source/'manifest.json')
    selected = read(source/'selection.json')['selected']
    samples = []
    custom = CustomEvaluator(metrics=['hit_rate','mrr'])
    for row in old:
        for variant in ['baseline',selected]:
            answer = read(source/'answers'/variant/(row['id']+'.json'))
            retrieval = read(source/variant/(row['id']+'.json'))
            assert answer['results'] == retrieval['results'][:5]
            pages = row['expected_pdf_pages']
            ir = {}
            if pages:
                for k in [5,10]:
                    scores = custom.evaluate(row['question'], [str(r['metadata']['page_num']) for r in retrieval['results'][:k]], ground_truth=[str(p) for p in pages])
                    ir.update({f'page_{name}_at_{k}':value for name,value in scores.items()})
            samples.append(dict(id=row['id'],variant=variant,split='dev' if row['id'] in split['dev'] else 'regression',
                positive=bool(pages), user_input=row['question'],response=answer['answer'],
                retrieved_contexts=[r['text'] for r in answer['results']], chunks=answer['results'],ir=ir))
    return samples

async def run(args, samples, settings):
    import httpx
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecisionWithoutReference
    recorder = Recorder(args.output)
    async with httpx.AsyncClient(timeout=180, event_hooks={'request':[recorder.request], 'response':[recorder.response]}) as client:
        evaluator = RagasEvaluator(settings, http_client=client, judge_max_tokens=2048, judge_temperature=0)
        llm, embeddings = evaluator._build_wrappers()
        metrics = dict(faithfulness=Faithfulness(llm=llm),answer_relevancy=AnswerRelevancy(llm=llm,embeddings=embeddings,strictness=3),
                       context_precision_without_reference=ContextPrecisionWithoutReference(llm=llm))
        semaphore = asyncio.Semaphore(args.workers)
        async def one(sample):
            async with semaphore:
                for name, metric in metrics.items():
                    path = args.output/'scores'/sample['variant']/sample['id']/(name+'.json')
                    fingerprint = hashed(sample)
                    if path.exists():
                        cached = read(path)
                        if cached['sample_sha256'] != fingerprint:
                            raise ValueError('Sample cache mismatch')
                        if cached['status'] in ('ok','undefined'):
                            continue
                    token = CURRENT.set(dict(question_id=sample['id'],variant=sample['variant'],metric=name))
                    start = time.monotonic()
                    for attempt in range(3):
                        try:
                            inputs = dict(user_input=sample['user_input'],response=sample['response'])
                            if name != 'answer_relevancy': inputs['retrieved_contexts'] = sample['retrieved_contexts']
                            result = await metric.ascore(**inputs)
                            score = finite(result.value)
                            record = dict(status='ok' if score is not None else 'undefined',value=score,reason=result.reason,
                                          undefined_note='Ragas returned no finite score (e.g. no factual claims).' if score is None else None)
                            break
                        except Exception as error:
                            if attempt < 2:
                                await asyncio.sleep(15*(attempt+1))
                                continue
                            record = dict(status='error',value=None,error_type=type(error).__name__)
                    record.update(id=sample['id'],variant=sample['variant'],metric=name,sample_sha256=fingerprint,seconds=time.monotonic()-start)
                    save(path,record)
                    CURRENT.reset(token)
                    print(f"{sample['id']} {sample['variant']} {name}: {record['status']} {record['value']}",flush=True)
        targets = samples[:2] if args.smoke else samples
        await asyncio.gather(*(one(s) for s in targets))

def report(output, samples):
    rows=[]
    for sample in samples:
        row={k:sample[k] for k in ['id','variant','split','positive','user_input','response','ir']}
        row['metrics'], row['status'] = {},{}
        for name in METRICS:
            path=output/'scores'/sample['variant']/sample['id']/(name+'.json')
            value=read(path) if path.exists() else dict(status='pending',value=None)
            row['metrics'][name],row['status'][name]=value['value'],value['status']
        rows.append(row)
    summary=dict(sample_n=len(rows),metric_results={state:sum(r['status'][m]==state for r in rows for m in METRICS) for state in ['ok','undefined','error','pending']},groups={})
    for split in ['all','dev','regression','negative']:
        summary['groups'][split]={}
        for variant in sorted({s['variant'] for s in samples}):
            group=[r for r in rows if r['variant']==variant and ((not r['positive']) if split=='negative' else (r['positive'] and (split=='all' or r['split']==split)))]
            summary['groups'][split][variant]=dict(n=len(group),**{m:mean_valid([r['metrics'][m] for r in group]) for m in METRICS})
            summary['groups'][split][variant]['ir']={m:mean_valid([r['ir'].get(m) for r in group]) for m in ['page_hit_rate_at_5','page_mrr_at_5','page_hit_rate_at_10','page_mrr_at_10']}
    summary['paired_positive_deltas']={}
    for metric in METRICS:
        deltas=[]
        for qid in {r['id'] for r in rows if r['positive']}:
            pair={r['variant']:r['metrics'][metric] for r in rows if r['id']==qid}
            other=next(v for v in pair if v!='baseline')
            if pair['baseline'] is not None and pair[other] is not None: deltas.append(pair[other]-pair['baseline'])
        summary['paired_positive_deltas'][metric]=mean_valid(deltas)
    calls=[json.loads(line) for line in (output/'api_calls.jsonl').read_text(encoding='utf-8').splitlines()] if (output/'api_calls.jsonl').exists() else []
    summary['api_calls']=len(calls)
    summary['tokens']={k:sum((c.get('usage') or {}).get(k,0) for c in calls) for k in ['prompt_tokens','completion_tokens','total_tokens']}
    summary['notes']=['Ragas LLM-as-judge, not human scores or answer accuracy.','Context precision uses generated response, not reference answer.',
        'NaN/None stored as null; errors/pending/undefined distinguished. Means exclude null with denominators reported.',
        'Correct abstention on negative questions can get low relevancy; negative results reported separately.',
        'Page IR metrics are page proxies, not exact gold chunk metrics.','Previously observed regression set; not an unseen holdout.',
        'Judge and generator both gpt-4o; evaluator bias and cross-language embedding effects remain.']
    save(output/'summary.json',summary)
    save(output/'all_results.json',rows)
    lines=['# Ragas 基线与优化全量评测','','三个指标均由Ragas真实调用模型计算，非人工评分。负例单列，未定义值不补零。', '',
           '| 指标（39道正例） | 基线均值 / 有效n | 优化均值 / 有效n |','|---|---|---|']
    variants=sorted(summary['groups']['all'])
    for metric in METRICS:
        values=[summary['groups']['all'][v][metric] for v in variants]
        lines.append('|'+metric+'|'+'|'.join(f"{v['mean']:.4f} / {v['valid_n']}" if v['mean'] is not None else 'N/A' for v in values)+'|')
    lines+=['','完整统计（含开发/回归/负例、同题配对差值与调用量）：','','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```']
    (output/'报告.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(metric_results=summary['metric_results'],groups=summary['groups']['all'],tokens=summary['tokens']),ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'data/textbook_ch1_3/experiments/20260922_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'data/textbook_ch1_3/experiments/20260922_ragas_v1')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--export-langfuse',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    args.output.mkdir(parents=True,exist_ok=True)
    samples=build_samples(args.source)
    settings=load_settings()
    manifest=dict(source=str(args.source),samples_sha256=hashed(samples),ragas_version=version('ragas'),
        judge_model=settings.llm.model,embedding_model=settings.embedding.model,temperature=0,judge_max_tokens=2048,
        metrics=METRICS,answer_relevancy_strictness=3,context_k=5,mode='score_saved_real_pipeline_outputs',
        note='Retrieval and generation are replayed from immutable source artifacts, not regenerated for judging.')
    if (args.output/'manifest.json').exists() and read(args.output/'manifest.json')!=manifest: raise ValueError('Manifest mismatch; use new output directory')
    save(args.output/'manifest.json',manifest)
    save(args.output/'samples.json',samples)
    if not args.report_only: asyncio.run(run(args,samples,settings))
    report(args.output,samples)
    from src.observability.langfuse_exporter import export_evaluation
    export = export_evaluation(samples, read(args.output/'all_results.json'), args.output.name,
                               enabled=args.export_langfuse)
    save(args.output/'langfuse_status.json', export)
    completion = read(args.output/'summary.json')['metric_results']
    if not args.smoke and not args.report_only and (completion['error'] or completion['pending']):
        raise SystemExit('Evaluation incomplete; inspect score statuses and resume the same command.')

if __name__=='__main__': main()
