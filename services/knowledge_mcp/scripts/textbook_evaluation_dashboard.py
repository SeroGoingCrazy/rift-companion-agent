"""Read-only Streamlit view of the saved paired Ragas experiment."""
import json
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'data/textbook_ch1_3/experiments/20260922_ragas_v1'
st.set_page_config(page_title='教材 RAG 评测', layout='wide')
st.title('教材 RAG：基线与优化对照')
st.caption('42道问题 · 两组真实回答 · Ragas自动评审。页命中率不等于答案正确率；回归集不是全新盲测。')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

if not (OUTPUT/'samples.json').exists():
    st.info('尚无评测样本。')
    st.stop()
samples=read(OUTPUT/'samples.json')

@st.fragment(run_every=15)
def progress():
    records=[read(p) for p in (OUTPUT/'scores').glob('*/*/*.json')]
    done=sum(r['status'] in ['ok','undefined'] for r in records)
    st.progress(done/(len(samples)*3), text=f'已完成 {done}/{len(samples)*3} 个指标计算')
    st.caption(f"未定义：{sum(r['status']=='undefined' for r in records)}；调用错误：{sum(r['status']=='error' for r in records)}。未定义值不会被替换成零分。")
progress()
@st.fragment(run_every=15)
def overview():
    records={(r['variant'],r['id'],r['metric']):r for r in
             [read(p) for p in (OUTPUT/'scores').glob('*/*/*.json')]}
    table=[]
    for split in ['all','dev','regression','negative']:
        for variant in sorted({s['variant'] for s in samples}):
            group=[s for s in samples if s['variant']==variant and
                   ((not s['positive']) if split=='negative' else
                    (s['positive'] and (split=='all' or s['split']==split)))]
            row={'范围':split,'方案':variant,'题数':len(group)}
            for metric in ['faithfulness','answer_relevancy','context_precision_without_reference']:
                values=[records.get((variant,s['id'],metric),{}).get('value') for s in group]
                values=[v for v in values if v is not None]
                row[metric]=sum(values)/len(values) if values else None
                row[metric+' 有效n']=len(values)
            table.append(row)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption('运行中均值为实时部分结果，以有效n为准。上下文精度采用无参考答案版本；负例正确拒答仍可能得到低相关性分数。')
overview()
qid=st.selectbox('逐题查看', sorted({s['id'] for s in samples}))
columns=st.columns(2)
for column,sample in zip(columns,[s for s in samples if s['id']==qid]):
    with column:
        st.subheader(sample['variant'])
        st.write(sample['user_input'])
        st.markdown(sample['response'])
        results=[read(p) for p in (OUTPUT/'scores'/sample['variant']/qid).glob('*.json')]
        st.dataframe([{'指标':r['metric'],'状态':r['status'],'分数':r['value']} for r in results],hide_index=True)
        for i,chunk in enumerate(sample['chunks'],1):
            with st.expander(f"[{i}] PDF {chunk['metadata']['page_num']} 页"):
                st.text(chunk['text'])
st.caption('Langfuse 接口已预留，本轮未连接云端项目。')
