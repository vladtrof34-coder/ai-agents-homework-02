"""Запуск этапов: python run_experiments.py retrieval|answers|memory|all."""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import homework02 as hw


def datasets():
    return (
        hw.read_jsonl(hw.DATA / "corpus.jsonl"),
        hw.read_jsonl(hw.DATA / "questions.jsonl"),
        hw.read_jsonl(hw.DATA / "unanswerable.jsonl"),
    )


def retrieval():
    pages, tasks, negatives = datasets()
    chars = []
    for page in pages:
        chars.extend(hw.chunk_chars(page))
    assert len(pages) >= 300 and len(chars) >= 300
    for task in tasks:
        assert any(
            p["pdf_page"] == task["pdf_page"] and task["evidence"] in p["text"]
            for p in pages
        ), task["id"]
    page_vecs = hw.embed_cached([hw.emb_text(c) for c in pages])
    char_vecs = hw.embed_cached([c["text"] for c in chars])
    queries = hw.embed_cached([t["question"] for t in tasks])
    rows, rankings = [], {}
    for name, chunks, vectors in [
        ("страницы", pages, page_vecs),
        ("400 символов", chars, char_vecs),
    ]:
        ranked = [hw.search_numpy(q, vectors, 20) for q in queries]
        rankings[name] = ranked
        rows.extend(
            {"нарезка": name, "k": k, "recall": hw.recall_of(ranked, chunks, tasks, k)}
            for k in [1, 3, 5, 10]
        )
    recall = pd.DataFrame(rows)
    hw.save_result("recall", recall.to_dict("records"))
    ax = recall.pivot(index="k", columns="нарезка", values="recall").plot(
        marker="o", ylim=(0, 1.05), grid=True
    )
    ax.set(ylabel="Recall@k", xlabel="k")
    ax.figure.savefig(hw.OUT / "recall.png", bbox_inches="tight", dpi=150)
    plt.close(ax.figure)
    words = []
    for task in tasks:
        words.append(hw.keyword_rank(task["question"], chars))
    vectors = rankings["400 символов"]
    hybrid = []
    for vector_result, word_result in zip(vectors, words):
        hybrid.append(hw.rrf([vector_result, word_result], 20))

    hybrid_rows = []
    for name, ranking in [
        ("вектор", vectors),
        ("слова", words),
        ("гибрид RRF", hybrid),
    ]:
        row = {"нарезка": "400 символов", "поиск": name}
        for k in [1, 3, 5, 10]:
            row[f"recall@{k}"] = hw.recall_of(ranking, chars, tasks, k)
        hybrid_rows.append(row)
    hybrid_table = pd.DataFrame(hybrid_rows)
    hw.save_result("hybrid", hybrid_rows)
    chosen = (
        recall[recall.k == 10].sort_values("recall", ascending=False).iloc[0]["нарезка"]
    )
    curve = recall[recall["нарезка"] == chosen]
    k = int(
        curve.loc[curve.recall >= curve.recall.max() - 1 / len(tasks) - 1e-9, "k"].min()
    )
    hw.K = k
    hw.save_result(
        "retrieval",
        {
            "pages": len(pages),
            "chars": len(chars),
            "k": k,
            "chosen": chosen,
            "criterion": "лучшая нарезка по Recall@10; минимальный k с потерей не более одного вопроса относительно её максимума",
        },
    )
    chunks, vectors = (
        (chars, char_vecs) if chosen == "400 символов" else (pages, page_vecs)
    )
    stats = hw.index_to_milvus(hw.COLLECTION, chunks, vectors)
    filtered = hw.client().search(
        hw.COLLECTION,
        data=[queries[0].tolist()],
        limit=3,
        filter="pdf_page <= 30",
        output_fields=["page", "pdf_page", "section", "text"],
    )[0]
    measured = {}
    for count in [1, 3, 5, 10]:
        found = hw.client().search(
            hw.COLLECTION,
            data=queries.tolist(),
            limit=count,
            output_fields=["page", "pdf_page", "text"],
        )
        correct = 0
        for hits, task in zip(found, tasks):
            for hit in hits:
                if hw.is_gold(hit["entity"], task):
                    correct += 1
                    break
        measured[str(count)] = correct / len(tasks)
    hw.save_result(
        "milvus",
        {
            "stats": stats,
            "filter": "pdf_page <= 30",
            "hits": [h["entity"] for h in filtered],
            "actual_recall": measured,
        },
    )
    print(recall.to_string(index=False), flush=True)
    print(hybrid_table.to_string(index=False), flush=True)
    print("Selected k:", k, "Indexed:", stats, flush=True)


def answers():
    _, tasks, negatives = datasets()
    hw.K = hw.RESULTS["retrieval"]["k"]
    configs = [
        ("strong_plain", hw.plain_answer, "strong"),
        ("cheap_plain", hw.plain_answer, "cheap"),
        ("cheap_rag", hw.rag_answer, "cheap"),
        ("cheap_agent", hw.agent, "cheap"),
        ("mid_rag", hw.rag_answer, "mid"),
    ]
    for name, fn, model in configs:
        print("Running", name, flush=True)
        frame = hw.evaluate(fn, tasks + negatives, name, hw.MODELS[model])
        print(
            name,
            "auto correct",
            frame.loc[frame.answerable, "correct_auto"].sum(),
            "/30",
            "refusals",
            frame.loc[~frame.answerable, "refused"].sum(),
            "/10",
            "USD",
            frame.cost.sum(),
            flush=True,
        )
    make_report()


def make_report():
    frames = [
        pd.DataFrame(hw.RESULTS[name])
        for name in [
            "strong_plain",
            "cheap_plain",
            "cheap_rag",
            "cheap_agent",
            "mid_rag",
        ]
    ]
    results = pd.concat(frames, ignore_index=True)
    results["correct"] = results.correct_auto
    if "manual_review" in hw.RESULTS:
        changes = hw.RESULTS["manual_review"]
        for item in changes:
            mask = (results.config == item["config"]) & (results.id == item["id"])
            results.loc[mask, "correct"] = item["correct"]
            if "refused" in item:
                results.loc[mask, "refused"] = item["refused"]
    table = hw.report(results)
    hw.save_result("report", table.reset_index().to_dict("records"))
    refusals = []
    for config, g in results.groupby("config", sort=False):
        refusals.append(
            {
                "config": config,
                "true_refusals": int(g.loc[~g.answerable, "refused"].sum()),
                "false_refusals": int(g.loc[g.answerable, "refused"].sum()),
            }
        )
    hw.save_result("refusals", refusals)
    fig, ax = plt.subplots(figsize=(8, 4))
    for config, r in table.iterrows():
        ax.scatter(r.cost_per_question * 100, r.accuracy * 100)
        ax.annotate(
            config,
            (r.cost_per_question * 100, r.accuracy * 100),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set(xlabel="Цена вопроса, центы USD", ylabel="Верных ответов, %", ylim=(-3, 105))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(hw.OUT / "money.png", dpi=150)
    plt.close(fig)
    print(table, flush=True)
    return results


def memory():
    hw.K = hw.RESULTS["retrieval"]["k"]
    mem = hw.Memory()
    mem.store({})
    sid = str(time.time_ns())
    first = []
    demo = []
    for question in [
        "Меня зовут Лена, я живу в Казани. Готовлюсь к вступительным экзаменам. Предпочитаю короткие ответы.",
        "Сколько вариантов в день рекомендует решать Ткачук?",
    ]:
        out = hw.talk(mem, sid + "-s1", first, question)
        demo.append({"session": 1, "question": question, **out})
    facts1 = mem.finish(first)
    second = []
    for question in [
        "Как меня зовут, где я живу и к чему готовлюсь?",
        "Я переехала: теперь живу в Перми, а не в Казани.",
    ]:
        out = hw.talk(mem, sid + "-s2", second, question)
        demo.append({"session": 2, "question": question, **out})
    facts2 = mem.finish(second)
    assert "Перм" in facts2.get("city", "") and "Казан" not in facts2.get(
        "city", ""
    ), facts2
    stored = mem.recall("В каком городе живёт пользователь?", 3)
    assert any("Перм" in t for t in stored) and not any(
        "Казан" in t for t in stored
    ), stored
    print("Memory sessions:", facts1, "->", facts2, flush=True)
    script = [
        "Какой объём домашних задач идёт после стандартного урока?",
        "Как автор советует проверять ответы?",
        "Что делать, если задача слишком трудная?",
        "Зачем записывать дату занятия?",
        "Как называется глава с краткими формулами?",
        "Сколько времени отводить на тренировочный вариант?",
        "Как меня зовут и какой формат ответов я предпочитаю?",
        "Я сейчас живу в Казани или в Перми?",
        "Сколько вариантов нужно для достоверной статистики?",
        "Дай короткий совет по подготовке с учётом моей цели.",
    ]
    bench = []
    for mode in ["full", "window"]:
        history = list(first + second) if mode == "full" else []
        for turn, question in enumerate(script, 1):
            out = hw.talk(mem, sid + "-bench-" + mode, history, question, mode)
            bench.append({"mode": mode, "turn": turn, "question": question, **out})
            print(mode, turn, "tokens", out["prompt"], flush=True)
    hw.save_result(
        "memory",
        {
            "demo": demo,
            "facts1": facts1,
            "facts2": facts2,
            "retrieved_after_update": stored,
            "benchmark": bench,
        },
    )
    frame = pd.DataFrame(bench)
    hw.save_result(
        "memory_summary",
        frame.groupby("mode")
        .agg(prompt=("prompt", "sum"), cost=("cost", "sum"))
        .reset_index()
        .to_dict("records"),
    )
    ax = frame.pivot(index="turn", columns="mode", values="prompt").plot(
        marker="o",
        grid=True,
        ylabel="Входные токены (все шаги агента)",
        xlabel="Реплика",
    )
    ax.figure.savefig(hw.OUT / "memory_tokens.png", bbox_inches="tight", dpi=150)
    plt.close(ax.figure)
    print(frame.groupby("mode")[["prompt", "cost"]].sum(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["retrieval", "answers", "memory", "report", "all"]
    )
    stage = parser.parse_args().stage

    if stage in ["retrieval", "all"]:
        retrieval()
    if stage in ["answers", "all"]:
        answers()
    if stage in ["memory", "all"]:
        memory()
    if stage == "report":
        make_report()
