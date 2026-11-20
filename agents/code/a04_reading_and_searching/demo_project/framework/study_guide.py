"""Build a lesson's study-guide PDF from its content file.

    python framework/study_guide.py how-llms-work v01 [--video path/to/rendered.mp4]

Reads <series>/study/<id>_study.py (LESSON dict + CONCEPTS list), extracts the referenced video frames into
<series>/study/img/ (only if missing, so the PDF can be rebuilt without the video), writes
<series>/study/<id>_study.html and prints it to <series>/study/<id>_study.pdf with headless Chrome.
"""
import argparse
import glob
import html
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIND_LABEL = {"mc": "Multiple choice", "tf": "True or false", "short": "Short answer",
              "number": "Calculation", "order": "Put in order", "code": "Hands-on code"}


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def load_content(series, vid):
    path = os.path.join(ROOT, series, "study", f"{vid}_study.py")
    spec = importlib.util.spec_from_file_location(f"{vid}_study", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LESSON, mod.CONCEPTS


def extract_frames(series, vid, concepts, video):
    img_dir = os.path.join(ROOT, series, "study", "img")
    os.makedirs(img_dir, exist_ok=True)
    wanted = {}
    for c in concepts:
        for fig in c.get("figures", []):
            fig["file"] = f"img/{vid}_{fig['t']:07.2f}.jpg"
            if not os.path.exists(os.path.join(ROOT, series, "study", fig["file"])):
                wanted[int(round(fig["t"] * 60))] = fig["file"]
    if not wanted:
        return
    if not video:
        found = glob.glob(os.path.join(ROOT, series, "media", "videos", f"{vid}_scene", "1080p60", "*.mp4"))
        if not found:
            sys.exit(f"missing frames {sorted(wanted.values())}: render the video first or pass --video")
        video = found[0]
    import av  # only needed when frames must be extracted
    with av.open(video) as container:
        fps = float(container.streams.video[0].average_rate)
        targets = {int(round(i / 60 * fps)): f for i, f in wanted.items()}
        for i, frame in enumerate(container.decode(video=0)):
            if i in targets:
                frame.to_image().resize((1280, 720)).save(os.path.join(ROOT, series, "study", targets[i]), quality=88)
                del targets[i]
                if not targets:
                    break
    if targets:
        sys.exit(f"video too short for frames {sorted(targets.values())}")


def render_exercise(num, ex):
    kind = KIND_LABEL[ex["kind"]]
    parts = [f'<div class="ex"><div class="exhead"><span class="exnum">{num}</span>'
             f'<span class="kind">{kind}</span></div><div class="q">{ex["q"]}</div>']
    if ex["kind"] == "mc":
        opts = "".join(f"<li><b>{chr(65 + i)}.</b> {o}</li>" for i, o in enumerate(ex["options"]))
        parts.append(f'<ol class="opts">{opts}</ol>')
    elif ex["kind"] == "tf":
        parts.append('<div class="tf">True &nbsp;&nbsp;☐ &nbsp;&nbsp;&nbsp;&nbsp; False &nbsp;&nbsp;☐</div>')
    elif ex["kind"] in ("short", "number", "order"):
        parts.append('<div class="line"></div>' * ex.get("lines", 1))
    if ex.get("code"):
        parts.append(f'<pre class="code">{html.escape(ex["code"])}</pre>')
    parts.append("</div>")
    return "".join(parts)


def build_html(lesson, concepts):
    n = len(concepts)
    out = [f"""<!doctype html><html><head><meta charset="utf-8"><title>{lesson['series']} · {lesson['label']} study guide</title>
<style>
@page {{ size: A4; margin: 14mm 15mm 15mm 15mm;
  @bottom-left {{ content: "{lesson['series']} · {lesson['label']} · {html.escape(lesson['title'])}"; font: 8pt 'DejaVu Sans'; color: #8a93a3; }}
  @bottom-right {{ content: "page " counter(page) " of " counter(pages); font: 8pt 'DejaVu Sans'; color: #8a93a3; }} }}
body {{ font: 10pt/1.42 'DejaVu Sans', sans-serif; color: #1d2433; margin: 0; }}
h1 {{ font: 700 25pt/1.2 'DejaVu Serif', serif; margin: 0 0 4pt; }}
h2 {{ font: 700 16pt/1.25 'DejaVu Serif', serif; margin: 2pt 0 5pt; }}
h3 {{ font: 700 11.5pt 'DejaVu Sans'; margin: 14pt 0 6pt; color: #1d2433; }}
code, pre {{ font-family: 'DejaVu Sans Mono', monospace; }}
code {{ background: #eef1f5; padding: 0 3px; border-radius: 3px; font-size: 9.5pt; }}
.kicker {{ font-size: 9pt; letter-spacing: .08em; text-transform: uppercase; color: #2a9d8f; font-weight: 700; }}
.sub {{ color: #56607a; font-size: 12pt; margin-bottom: 14pt; }}
.chip {{ display: inline-block; background: #1d2433; color: #fff; border-radius: 10px; padding: 1px 9px; font-size: 8.5pt; }}
.chip.play::before {{ content: "▶ "; color: #f4c95d; }}
.concept {{ break-before: page; }}
.box {{ border-radius: 6px; padding: 7pt 11pt; margin: 7pt 0; break-inside: avoid; }}
.key {{ background: #e7f5f3; border-left: 4px solid #2a9d8f; }}
.key b.t {{ color: #1f7a6f; display: block; font-size: 9pt; letter-spacing: .06em; text-transform: uppercase; margin-bottom: 2pt; }}
.rule {{ background: #fff7e0; border-left: 4px solid #e0a800; }}
.note {{ background: #f2f4f8; border-left: 4px solid #8a93a3; font-size: 9.5pt; }}
figure {{ margin: 6pt auto; width: 62%; break-inside: avoid; }}
figure.small {{ width: 48%; }}
.figrow {{ display: flex; gap: 10pt; }} .figrow figure {{ width: auto; flex: 1; margin: 8pt 0; }}
figure img {{ width: 100%; border-radius: 6px; display: block; }}
figcaption {{ font-size: 9pt; color: #56607a; margin-top: 4pt; }}
pre.code {{ background: #272822; color: #f8f8f2; font-size: 8.8pt; line-height: 1.4; padding: 9pt 11pt; border-radius: 6px;
  white-space: pre-wrap; break-inside: avoid; margin: 6pt 0; }}
.check {{ border: 1.5px solid #2a9d8f; border-radius: 8px; padding: 8pt 12pt 2pt; margin-top: 10pt; }}
.check > .ctitle {{ font-weight: 700; color: #1f7a6f; margin-bottom: 2pt; }}
.check > .cnote {{ font-size: 9pt; color: #56607a; margin-bottom: 6pt; }}
.ex {{ margin: 0 0 8pt; break-inside: avoid; }}
.exhead {{ margin-bottom: 1pt; }}
.exnum {{ font-weight: 700; color: #2a9d8f; margin-right: 8pt; }}
.kind {{ font-size: 8pt; text-transform: uppercase; letter-spacing: .06em; color: #8a93a3; }}
.opts {{ list-style: none; padding-left: 14pt; margin: 3pt 0 0; }}
.opts li {{ margin: 1pt 0; }}
.tf {{ padding-left: 14pt; margin-top: 3pt; font-size: 10pt; color: #56607a; }}
.line {{ border-bottom: 1px dotted #b9c0cc; height: 16pt; margin-left: 14pt; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9.8pt; margin: 8pt 0; }}
th, td {{ text-align: left; padding: 5pt 7pt; border-bottom: 1px solid #dde2ea; vertical-align: top; }}
th {{ font-size: 8.5pt; text-transform: uppercase; letter-spacing: .05em; color: #56607a; }}
td.box1 {{ font-size: 13pt; color: #8a93a3; text-align: center; }}
.answers {{ break-before: page; }}
.ans {{ margin: 0 0 7pt; break-inside: avoid; }}
.ans .a {{ font-weight: 700; }}
.ans .why {{ color: #3b4457; }}
.ans .re {{ font-size: 8.5pt; color: #8a93a3; }}
.agroup {{ margin-top: 12pt; break-inside: avoid-page; }}
</style></head><body>"""]

    # Cover page: title, how to use, concept map
    rows = "".join(
        f"<tr><td><b>{i}.</b> {html.escape(c['title'])}</td><td><span class='chip play'>{mmss(c['segment'][0])}–{mmss(c['segment'][1])}</span></td>"
        f"<td>{len(c['exercises'])}</td><td class='box1'>☐</td></tr>" for i, c in enumerate(concepts, 1))
    out.append(f"""
<div class="kicker">{lesson['series']} · {lesson['label']} · Study guide</div>
<h1>{html.escape(lesson['title'])}</h1>
<div class="sub">{html.escape(lesson['tagline'])} · video {lesson['duration']}</div>
{lesson['intro']}
<div class="box rule"><b>How to use this guide: master one concept before moving to the next.</b>
<ol style="margin:4pt 0 0 0; padding-left: 16pt">
<li>Watch the whole video once, without stopping.</li>
<li>Then take the concepts <b>in order</b>. For each one: read the short explanation, look at the picture from the video
(▶ is the time in the video), and answer every question in its <i>Check</i> without looking anything up.</li>
<li>Check your answers in the <b>Answers</b> section at the end (page {{answers_page}}).</li>
<li><b>Pass mark: every question in the check correct.</b> If you missed one, rewatch that concept's video segment,
reread the explanation, and redo only the questions you missed. Tick the concept off below, then move on.</li>
</ol>
<div style="margin-top:4pt">It is strongly suggested that you <b>master each concept before starting the next one</b>. Each concept builds on the previous ones.</div></div>
<h3>Concepts in this lesson</h3>
<table><tr><th>Concept</th><th>Video segment</th><th>Questions</th><th>Mastered</th></tr>{rows}</table>
{lesson.get('prereq', '')}""")

    # Concepts
    qnum = {}
    for i, c in enumerate(concepts, 1):
        parts = [f'<section class="concept"><div class="kicker">Concept {i} of {n}</div><h2>{html.escape(c["title"])}</h2>'
                 f'<div style="margin-bottom:6pt"><span class="chip play">Video {mmss(c["segment"][0])}–{mmss(c["segment"][1])}</span></div>']
        for block in c["body"]:
            parts.append(block)
        parts.append(f'<div class="check"><div class="ctitle">Check {i}: {html.escape(c["title"])}</div>'
                     f'<div class="cnote">Answer all {len(c["exercises"])} questions before looking at the answers. '
                     f'Pass mark: all correct.</div>')
        for j, ex in enumerate(c["exercises"], 1):
            qnum[(i, j)] = f"{i}.{j}"
            parts.append(render_exercise(f"{i}.{j}", ex))
        parts.append("</div></section>")
        out.append("".join(parts))

    # Answers
    out.append('<section class="answers"><div class="kicker">Answers</div><h2>Answers and explanations</h2>'
               '<p>Mark each answer right or wrong. If a concept has any wrong answer, rewatch its segment (▶) '
               'and redo the questions you missed before moving on.</p>')
    for i, c in enumerate(concepts, 1):
        out.append(f'<div class="agroup"><h3>{i}. {html.escape(c["title"])} '
                   f'<span class="re">· rewatch ▶ {mmss(c["segment"][0])}–{mmss(c["segment"][1])}</span></h3>')
        for j, ex in enumerate(c["exercises"], 1):
            out.append(f'<div class="ans"><span class="exnum">{i}.{j}</span><span class="a">{ex["answer"]}</span> '
                       f'<span class="why">{ex.get("why", "")}</span></div>')
        out.append("</div>")
    out.append("</section></body></html>")
    return "\n".join(out)


def print_pdf(html_path, pdf_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or "/opt/pw-browsers/chromium"
    with tempfile.TemporaryDirectory() as profile:  # own profile, so several builds can run at once
        subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", f"--print-to-pdf={pdf_path}",
                        "file://" + os.path.abspath(html_path)], check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series")
    ap.add_argument("video_id")
    ap.add_argument("--video", help="rendered mp4 to take frames from (default: <series>/media/...)")
    args = ap.parse_args()
    lesson, concepts = load_content(args.series, args.video_id)
    extract_frames(args.series, args.video_id, concepts, args.video)
    for c in concepts:  # body blocks place figures with the placeholders {fig0}, {fig1}, ...
        for k, fig in enumerate(c.get("figures", [])):
            c["body"] = [b.replace("{fig%d}" % k, figure_html(fig)) for b in c["body"]]
    study = os.path.join(ROOT, args.series, "study")
    html_path = os.path.join(study, f"{args.video_id}_study.html")
    pdf_path = os.path.join(study, f"{args.video_id}_study.pdf")
    doc = build_html(lesson, concepts)
    # Two passes: find the page where the answers start, then write it into the instructions.
    open(html_path, "w").write(doc.replace("{answers_page}", "?"))
    print_pdf(html_path, pdf_path)
    page = answers_start_page(pdf_path)
    open(html_path, "w").write(doc.replace("{answers_page}", str(page)))
    print_pdf(html_path, pdf_path)
    print(f"wrote {os.path.relpath(pdf_path, ROOT)} (answers start on page {page})")


def figure_html(fig):
    cls = f' class="{fig["size"]}"' if fig.get("size") else ""
    return (f'<figure{cls}><img src="{fig["file"]}"><figcaption><span class="chip play">{mmss(fig["t"])}</span> '
            f'{fig["caption"]}</figcaption></figure>')


def answers_start_page(pdf_path):
    from pypdf import PdfReader
    for i, page in enumerate(PdfReader(pdf_path).pages, 1):
        if "Answers and explanations" in (page.extract_text() or ""):
            return i
    return "?"


if __name__ == "__main__":
    main()
