"""Запуск этапов: python run_experiments.py retrieval|answers|memory|all."""
import argparse
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import homework02 as hw


def datasets():
    return (hw.read_jsonl(hw.DATA / 'corpus.jsonl'), hw.read_jsonl(hw.DATA / 'questions.jsonl'),
            hw.read_jsonl(hw.DATA / 'unanswerable.jsonl'))


def retrieval():
    pages, tasks, negatives = datasets()
    chars = [c for p in pages for c in hw.chunk_chars(p)]
    assert len(pages) >= 300 and len(chars) >= 300
    for task in tasks:
        assert any(p['pdf_page'] == task['pdf_page'] and task['evidence'] in p['text'] for p in pages), task['id']
    page_vecs = hw.embed_cached([hw.emb_text(c) for c in pages])
    char_vecs = hw.embed_cached([c['text'] for c in chars])
    queries = hw.embed_cached([t['question'] for t in tasks])
    rows, rankings = [], {}
    for name, chunks, vectors in [('страницы', pages, page_vecs), ('400 символов', chars, char_vecs)]:
        ranked = [hw.search_numpy(q, vectors, 20) for q in queries]
        rankings[name] = ranked
        rows.extend({'нарезка': name, 'k': k, 'recall': hw.recall_of(ranked, chunks, tasks, k)} for k in [1, 3, 5, 10])
    recall = pd.DataFrame(rows)
    recall.to_csv(hw.OUT / 'recall.csv', index=False)
    ax = recall.pivot(index='k', columns='нарезка', values='recall').plot(marker='o', ylim=(0, 1.05), grid=True)
    ax.set(ylabel='Recall@k', xlabel='k')
    ax.figure.savefig(hw.OUT / 'recall.png', bbox_inches='tight', dpi=150)
    plt.close(ax.figure)
    hybrid_rows=[]
    for cut,chunks in [('страницы',pages),('400 символов',chars)]:
        words=[hw.keyword_rank(t['question'],chunks) for t in tasks]
        hybrid=[hw.rrf([a,b],20) for a,b in zip(rankings[cut],words)]
        for name,ranks in [('вектор',rankings[cut]),('слова',words),('гибрид RRF',hybrid)]:
            hybrid_rows.append({'нарезка':cut,'поиск':name,**{f'recall@{k}':hw.recall_of(ranks,chunks,tasks,k) for k in [1,3,5,10]}})
    hybrid_table=pd.DataFrame(hybrid_rows)
    hybrid_table.to_csv(hw.OUT/'hybrid.csv',index=False)
    chosen=recall[recall.k==10].sort_values('recall',ascending=False).iloc[0]['нарезка']
    curve=recall[recall['нарезка']==chosen]
    k=int(curve.loc[curve.recall >= curve.recall.max()-1/len(tasks)-1e-9,'k'].min())
    hw.K=k
    hw.write_json(hw.OUT/'retrieval.json',{'pages':len(pages),'chars':len(chars),'k':k,'chosen':chosen,
        'criterion':'лучшая нарезка по Recall@10; минимальный k с потерей не более одного вопроса относительно её максимума',
        'page_rankings':rankings['страницы']})
    chunks,vectors=(chars,char_vecs) if chosen=='400 символов' else (pages,page_vecs)
    stats=hw.index_to_milvus(hw.COLLECTION,chunks,vectors)
    filtered=hw.client().search(hw.COLLECTION,data=[queries[0].tolist()],limit=3,filter='pdf_page <= 30',
        output_fields=['page','pdf_page','section','text'])[0]
    measured={}
    for count in [1,3,5,10]:
        found=hw.client().search(hw.COLLECTION,data=queries.tolist(),limit=count,output_fields=['page','pdf_page','text'])
        measured[str(count)]=float(np.mean([any(hw.is_gold(h['entity'],t) for h in row) for row,t in zip(found,tasks)]))
    hw.write_json(hw.OUT/'milvus.json',{'stats':stats,'filter':'pdf_page <= 30','hits':[h['entity'] for h in filtered], 'actual_recall':measured})
    print(recall.to_string(index=False),flush=True)
    print(hybrid_table.to_string(index=False),flush=True)
    print('Selected k:',k,'Indexed:',stats,flush=True)


def answers():
    _,tasks,negatives=datasets()
    hw.K=json.loads((hw.OUT/'retrieval.json').read_text())['k']
    configs=[('strong_plain',hw.plain_answer,'strong'),('cheap_plain',hw.plain_answer,'cheap'),
             ('cheap_rag',hw.rag_answer,'cheap'),('cheap_agent',hw.agent,'cheap'),('mid_rag',hw.rag_answer,'mid')]
    for name,fn,model in configs:
        print('Running',name,flush=True)
        frame=hw.evaluate(fn,tasks+negatives,name,hw.MODELS[model])
        print(name,'auto correct',frame.loc[frame.answerable,'correct_auto'].sum(),'/30',
            'refusals',frame.loc[~frame.answerable,'refused'].sum(),'/10','USD',frame.cost.sum(),flush=True)
    make_report()


def make_report():
    frames=[pd.DataFrame(hw.read_jsonl(hw.OUT/f'{name}.jsonl')) for name in
            ['strong_plain','cheap_plain','cheap_rag','cheap_agent','mid_rag']]
    results=pd.concat(frames,ignore_index=True)
    results['correct']=results.correct_auto
    review=hw.OUT/'manual_review.json'
    if review.exists():
        changes=json.loads(review.read_text())
        for item in changes:
            mask=(results.config==item['config']) & (results.id==item['id'])
            results.loc[mask,'correct']=item['correct']
            if 'refused' in item:
                results.loc[mask,'refused']=item['refused']
    results.drop(columns=['hits','trace'],errors='ignore').to_csv(hw.OUT/'answers.csv',index=False)
    table=hw.report(results)
    table.to_csv(hw.OUT/'report.csv')
    refusals=[]
    for config,g in results.groupby('config',sort=False):
        refusals.append({'config':config,'true_refusals':int(g.loc[~g.answerable,'refused'].sum()),
                         'false_refusals':int(g.loc[g.answerable,'refused'].sum())})
    pd.DataFrame(refusals).to_csv(hw.OUT/'refusals.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4))
    for config,r in table.iterrows():
        ax.scatter(r.cost_per_question*100,r.accuracy*100)
        ax.annotate(config,(r.cost_per_question*100,r.accuracy*100),xytext=(4,4),textcoords='offset points',fontsize=9)
    ax.set(xlabel='Цена вопроса, центы USD',ylabel='Верных ответов, %',ylim=(-3,105))
    ax.grid(alpha=.3);fig.tight_layout();fig.savefig(hw.OUT/'money.png',dpi=150);plt.close(fig)
    rag=results[results.answerable & results.config.isin(['cheap_rag','cheap_agent','mid_rag'])]
    best=rag.groupby('config').correct.mean().idxmax()
    failures=rag[(rag.config==best)&~rag.correct]
    hw.write_json(hw.OUT/'failures.json',{'config':best,'errors':failures.to_dict('records')})
    print(table,flush=True)
    return results


def memory():
    hw.K=json.loads((hw.OUT/'retrieval.json').read_text())['k']
    mem=hw.Memory()
    mem.store({})
    sid=str(time.time_ns())
    first=[];demo=[]
    for question in ['Меня зовут Лена, я живу в Казани. Готовлюсь к вступительным экзаменам. Предпочитаю короткие ответы.',
                     'Сколько вариантов в день рекомендует решать Ткачук?']:
        out=hw.talk(mem,sid+'-s1',first,question)
        demo.append({'session':1,'question':question,**out})
    facts1=mem.finish(first)
    second=[]
    for question in ['Как меня зовут, где я живу и к чему готовлюсь?',
                     'Я переехала: теперь живу в Перми, а не в Казани.']:
        out=hw.talk(mem,sid+'-s2',second,question)
        demo.append({'session':2,'question':question,**out})
    facts2=mem.finish(second)
    assert 'Перм' in facts2.get('city','') and 'Казан' not in facts2.get('city',''),facts2
    stored=mem.recall('В каком городе живёт пользователь?',3)
    assert any('Перм' in t for t in stored) and not any('Казан' in t for t in stored),stored
    print('Memory sessions:',facts1,'->',facts2,flush=True)
    script=['Какой объём домашних задач идёт после стандартного урока?',
            'Как автор советует проверять ответы?',
            'Что делать, если задача слишком трудная?',
            'Зачем записывать дату занятия?',
            'Как называется глава с краткими формулами?',
            'Сколько времени отводить на тренировочный вариант?',
            'Как меня зовут и какой формат ответов я предпочитаю?',
            'Я сейчас живу в Казани или в Перми?',
            'Сколько вариантов нужно для достоверной статистики?',
            'Дай короткий совет по подготовке с учётом моей цели.']
    bench=[]
    for mode in ['full','window']:
        history=list(first+second) if mode=='full' else []
        for turn,question in enumerate(script,1):
            out=hw.talk(mem,sid+'-bench-'+mode,history,question,mode)
            bench.append({'mode':mode,'turn':turn,'question':question,**out})
            print(mode,turn,'tokens',out['prompt'],flush=True)
    hw.write_json(hw.OUT/'memory.json',{'demo':demo,'facts1':facts1,'facts2':facts2,'retrieved_after_update':stored,'benchmark':bench})
    frame=pd.DataFrame(bench)
    frame[['mode','turn','prompt','cost']].to_csv(hw.OUT/'memory_tokens.csv',index=False)
    frame.groupby('mode').agg(prompt=('prompt','sum'),cost=('cost','sum')).to_csv(hw.OUT/'memory_summary.csv')
    ax=frame.pivot(index='turn',columns='mode',values='prompt').plot(marker='o',grid=True,ylabel='Входные токены (все шаги агента)',xlabel='Реплика')
    ax.figure.savefig(hw.OUT/'memory_tokens.png',bbox_inches='tight',dpi=150);plt.close(ax.figure)
    print(frame.groupby('mode')[['prompt','cost']].sum(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['retrieval','answers','memory','report','all'])
    stage=parser.parse_args().stage
    for name,fn in [('retrieval',retrieval),('answers',answers),('memory',memory),('report',make_report)]:
        if stage==name or (stage=='all' and name!='report'):
            fn()
