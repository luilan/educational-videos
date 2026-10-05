"""LLMs in Practice, episode 4 — Embeddings for Search: Meaning as Distance.

Render from the repo root:  ./render.sh llms-in-practice p04
Every vector value, 2-D position and score on screen comes from code/p04_embedding_search/search.py
(sentence-transformers/all-MiniLM-L6-v2, exact PCA via SVD).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from p04_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "LLMs IN PRACTICE  ·  EPISODE 4"
# (short label, PCA x, PCA y, group)
DOCS = [("boil eggs 10 min", -0.49, -0.14, "egg"), ("poached egg 3 min", -0.53, -0.10, "egg"),
        ("cats love boxes", 0.44, -0.54, "cat"), ("dogs need walks", 0.48, 0.12, "dog"),
        ("preheat the oven", -0.27, 0.31, "oven"), ("train to Milan", 0.28, 0.73, "train"),
        ("store cooked eggs", -0.47, -0.15, "egg"), ("cat naps on sofa", 0.56, -0.23, "cat")]
KITTEN = (0.44, -0.25)
EGG_Q = (-0.41, -0.06)
KITTEN_SCORES = [0.18, 0.11, 0.60, 0.21, 0.03, 0.18, 0.15, 0.65]
GROUP_COLORS = {"egg": GOLD, "cat": TEAL_C, "dog": GREEN_C, "oven": RED_C, "train": BLUE_C}
VECTOR = ["0.082", "0.038", "-0.003", "0.083", "-0.020"]
CODE = """doc_vectors = embed(documents)          # once, then store them

def search(question, k=3):
    q = embed([question])[0]
    scores = doc_vectors @ q             # cosine similarity
    return topk(scores, k)"""
PLOT_SCALE = 4.2


def pos(x, y, center=(0.3, -0.2)):
    return [center[0] + x * PLOT_SCALE, center[1] + y * PLOT_SCALE, 0]


def note(text, color=GREY_D, font_size=22):
    label = Text(text, font_size=font_size)
    box = RoundedRectangle(corner_radius=0.12, width=label.width + 0.5, height=label.height + 0.35, stroke_width=0,
                           fill_color=color, fill_opacity=0.9)
    return VGroup(box, label.move_to(box))


class EmbeddingSearchVideo(VoicedScene):
    VIDEO = "p04"

    def construct(self):
        play_token_intro(self, TITLE, 4, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.embedding()     # 2
        self.how()           # 3
        self.map()           # 4
        self.search()        # 5
        self.more()          # 6
        self.code()          # 7
        self.scale()         # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        box = RoundedRectangle(corner_radius=0.3, width=6, height=0.8, stroke_color=GREY_B)
        q = Text("kitten", font_size=30).move_to(box).align_to(box, LEFT).shift(0.4 * RIGHT)
        bar = VGroup(box, q).move_to([0, 2.4, 0])
        self.at("kitten")
        self.play(Create(box), Write(q), run_time=0.6)
        none = Text("0 results", font_size=28, color=RED_B).next_to(bar, DOWN, buff=0.3)
        self.at("nothing")
        self.play(FadeIn(none), run_time=0.4)
        notes = VGroup(note("Cats love sleeping in cardboard boxes."), note("Our cat naps on the sofa all afternoon."))
        notes.arrange(DOWN, buff=0.3).move_to([0, -0.4, 0])
        self.at("cat")
        self.play(LaggedStart(*[FadeIn(n, shift=0.2 * UP) for n in notes], lag_ratio=0.3), run_time=0.7)
        letters = Text("keyword search: letters, not meaning", font_size=28, color=GREY_A).move_to([0, -2.2, 0])
        self.at("letters")
        self.play(FadeIn(letters), run_time=0.4)
        fix = Text("embeddings", font_size=40, color=YELLOW).move_to([0, -3.1, 0])
        self.at("embeddings")
        self.play(FadeIn(fix, scale=1.2), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. what an embedding is
    def embedding(self):
        self.section(2)
        self.clear_stage()
        text = note("Cats love sleeping in cardboard boxes.").scale(0.75).move_to([-4.5, 1.8, 0])
        model = VGroup(RoundedRectangle(corner_radius=0.2, width=2.6, height=1.3, stroke_color=MODEL_COLOR,
                                        fill_color=MODEL_COLOR, fill_opacity=0.25),
                       Text("embedding\nmodel", font_size=24, line_spacing=0.8)).move_to([-0.6, 1.8, 0])
        model[1].move_to(model[0])
        nums = VGroup(*[Text(v, font=MONO, font_size=22) for v in VECTOR], Text("⋮", font_size=26))
        nums.arrange(DOWN, buff=0.12)
        bracket = SurroundingRectangle(nums, buff=0.15, color=GREY_B, corner_radius=0.05)
        vec = VGroup(bracket, nums).move_to([2.6, 1.8, 0])
        a1 = Arrow(text.get_right(), model.get_left(), buff=0.15, color=GREY_B)
        a2 = Arrow(model.get_right(), vec.get_left(), buff=0.15, color=GREY_B)
        self.at("reads")
        self.play(FadeIn(text), GrowArrow(a1), FadeIn(model), run_time=0.7)
        self.at("vector")
        self.play(GrowArrow(a2), FadeIn(vec, shift=0.2 * LEFT), run_time=0.6)
        size = Text("384 numbers", font_size=28, color=YELLOW).next_to(vec, RIGHT, buff=0.4)
        self.at("384")
        self.play(FadeIn(size), run_time=0.4)

        origin = [-1.5, -2.4, 0]
        cats1 = Arrow(origin, [1.8, -1.2, 0], buff=0, color=TEAL_C, stroke_width=5)
        cats2 = Arrow(origin, [1.9, -1.7, 0], buff=0, color=TEAL_C, stroke_width=5)
        train = Arrow(origin, [-0.9, -0.3, 0], buff=0, color=BLUE_C, stroke_width=5)
        lab = VGroup(Text("two cat sentences", font_size=20, color=TEAL_C).next_to(cats1.get_end(), RIGHT, 0.15),
                     Text("train to Milan", font_size=20, color=BLUE_C).next_to(train.get_end(), LEFT, 0.15))
        self.at("similar")
        self.play(GrowArrow(cats1), GrowArrow(cats2), FadeIn(lab[0]), run_time=0.7)
        self.at("directions")
        self.play(GrowArrow(train), FadeIn(lab[1]), run_time=0.5)
        same = Text("like token embeddings (How LLMs Work, ep. 3), for whole sentences", font_size=22,
                    color=GREY_B).to_edge(DOWN, buff=0.35)
        self.at("token")
        self.play(FadeIn(same), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 3. how it is made
    def how(self):
        self.section(3)
        self.clear_stage()
        toks = VGroup(*[token(w, font_size=22) for w in ["Cats", "love", "sleeping", "in", "boxes"]])
        toks.arrange(RIGHT, buff=0.25).move_to([0, 2.5, 0])
        tf = VGroup(RoundedRectangle(corner_radius=0.2, width=7.5, height=0.9, stroke_color=MODEL_COLOR,
                                     fill_color=MODEL_COLOR, fill_opacity=0.25),
                    Text("small transformer", font_size=24)).move_to([0, 1.2, 0])
        tf[1].move_to(tf[0])
        self.at("transformer")
        self.play(FadeIn(toks), FadeIn(tf), run_time=0.6)
        cols = VGroup(*[Rectangle(width=0.35, height=1.0, stroke_width=0, fill_color=TOKEN_COLOR, fill_opacity=0.8)
                        for _ in toks])
        for c, t in zip(cols, toks):
            c.move_to([t.get_x(), -0.2, 0])
        self.at("each")
        self.play(LaggedStart(*[GrowFromEdge(c, UP) for c in cols], lag_ratio=0.1), run_time=0.7)
        avg = Rectangle(width=0.5, height=1.0, stroke_width=0, fill_color=YELLOW, fill_opacity=0.85).move_to([0, -2.0, 0])
        avg_l = Text("average", font_size=22, color=YELLOW).next_to(avg, LEFT, buff=0.3)
        self.at("average")
        self.play(*[ReplacementTransform(c.copy(), avg) for c in cols], FadeIn(avg_l), run_time=0.8)
        circle = Circle(radius=0.9, color=GREY_C).move_to([3.6, -2.0, 0])
        unit = Arrow(circle.get_center(), circle.point_at_angle(0.6), buff=0, color=YELLOW, stroke_width=5)
        unit_l = Text("length 1: only direction matters", font_size=22, color=GREY_A).next_to(circle, RIGHT, 0.2)
        unit_l.shift(0.2 * LEFT).next_to(circle, DOWN, buff=0.15)
        self.at("scale")
        self.play(Create(circle), GrowArrow(unit), FadeIn(unit_l), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 4. the map
    def map(self):
        self.section(4)
        self.clear_stage()
        title = Text("384 dimensions → 2 (real positions)", font_size=26, color=GREY_B).to_edge(UP, buff=0.35)
        self.at("squash")
        self.play(FadeIn(title, shift=0.2 * DOWN), run_time=0.5)
        self.dots = VGroup()
        self.labels = VGroup()
        for label, x, y, g in DOCS:
            d = Dot(pos(x, y), radius=0.11, color=GROUP_COLORS[g])
            self.dots.add(d)
        offsets = {0: UP + 0.6 * RIGHT, 1: UP + 0.9 * LEFT, 6: DOWN, 2: DOWN, 7: RIGHT, 3: RIGHT, 4: UP, 5: RIGHT}
        for i, (label, x, y, g) in enumerate(DOCS):
            self.labels.add(Text(label, font_size=20, color=GROUP_COLORS[g]).next_to(self.dots[i], offsets[i], 0.2))
        self.at("real")
        self.play(LaggedStart(*[FadeIn(VGroup(d, l), scale=0.6) for d, l in zip(self.dots, self.labels)],
                              lag_ratio=0.1), run_time=1.2)
        for cue, idx in [("eggs", [0, 1, 6]), ("cats", [2, 7]), ("milan", [5])]:
            ring = SurroundingRectangle(VGroup(*[self.dots[i] for i in idx], *[self.labels[i] for i in idx]),
                                        buff=0.15, corner_radius=0.2, color=GROUP_COLORS[DOCS[idx[0]][3]])
            self.at(cue)
            self.play(Create(ring), run_time=0.5)
            self.play(FadeOut(ring), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5. search
    def search(self):
        self.section(5)
        self.q = Star(n=5, outer_radius=0.2, color=YELLOW, fill_opacity=1).move_to(pos(*KITTEN))
        q_label = Text("“Where does my kitten like to sleep?”", font_size=22, color=YELLOW)
        q_label.next_to(self.q, DOWN, buff=0.25)
        self.at("question")
        self.play(FadeIn(self.q, scale=2), run_time=0.5)
        self.at("kitten")
        self.play(FadeIn(q_label), run_time=0.4)
        self.at("next")
        self.play(Indicate(self.dots[2]), Indicate(self.dots[7]), run_time=0.7)
        map_group = VGroup(self.dots, self.labels, self.q, q_label)
        self.at("rank")
        self.play(map_group.animate.scale(0.72).move_to([-2.35, -0.3, 0]), run_time=0.7)
        formula = Text("score = cosine similarity = q · d", font=MONO, font_size=22).move_to([3.4, 2.4, 0])
        self.at("cosine")
        self.play(FadeIn(formula), run_time=0.5)
        order = sorted(range(8), key=lambda i: -KITTEN_SCORES[i])
        rows = VGroup(*[VGroup(Text(f"{KITTEN_SCORES[i]:.2f}", font=MONO, font_size=24),
                               Text(DOCS[i][0], font_size=24, color=GROUP_COLORS[DOCS[i][3]])).arrange(RIGHT, buff=0.4)
                        for i in order[:4]]).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.4, 0.4, 0])
        for k, cue in enumerate(["sofa", "boxes", "dogs"]):
            self.at(cue)
            self.play(FadeIn(rows[k], shift=0.2 * LEFT), run_time=0.4)
        none = Text("0 words in common", font_size=30, color=YELLOW).move_to([3.4, -1.9, 0])
        self.at("common")
        self.play(FadeIn(none, scale=1.2), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 6. more queries
    def more(self):
        self.section(6)
        self.clear_stage()
        results = [("“How long should I cook a hard-boiled egg?”", [("0.65", "Boil eggs for 10 minutes for firm yolks."),
                                                                   ("0.57", "Store cooked eggs in the fridge …"),
                                                                   ("0.51", "Simmer a poached egg for about 3 minutes.")]),
                   ("“When is the first departure to Milan?”", [("0.57", "The train to Milan leaves at 8 in the morning."),
                                                               ("0.10", "Preheat the oven to 200 degrees …"),
                                                               ("0.10", "Our cat naps on the sofa …")])]
        blocks = VGroup()
        for q, rows in results:
            head = Text(q, font_size=26, color=YELLOW)
            body = VGroup(*[VGroup(Text(s, font=MONO, font_size=22), Text(d, font_size=22)).arrange(RIGHT, buff=0.35)
                            for s, d in rows]).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            blocks.add(VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.25))
        blocks.arrange(DOWN, aligned_edge=LEFT, buff=0.7).move_to([0, 0.2, 0])
        self.at("hard")
        self.play(FadeIn(blocks[0][0]), run_time=0.4)
        self.at("best")
        self.play(FadeIn(blocks[0][1], shift=0.2 * UP), blocks[0][1][0].animate.set_color(GREEN_B), run_time=0.6)
        self.at("milan")
        self.play(FadeIn(blocks[1][0]), run_time=0.4)
        self.at("wins")
        self.play(FadeIn(blocks[1][1], shift=0.2 * UP), blocks[1][1][0].animate.set_color(GREEN_B), run_time=0.6)
        self.at("zero")
        self.play(Indicate(VGroup(blocks[1][1][1][0], blocks[1][1][2][0]), color=RED), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 7. code
    def code(self):
        self.section(7)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("once")
        self.play(Create(hl), run_time=0.4)
        self.at("question")
        self.play(highlight(hl, code, 3), run_time=0.4)
        self.at("dot")
        self.play(highlight(hl, code, 4), run_time=0.4)
        self.at("top")
        self.play(highlight(hl, code, 5), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. at scale, and limits
    def scale(self):
        self.section(8)
        self.clear_stage()
        import random
        rng = random.Random(4)
        cloud = VGroup(*[Dot([rng.uniform(-6, -0.8), rng.uniform(-2.6, 2.4), 0], radius=0.04, color=GREY_B)
                         for _ in range(260)])
        db = Text("vector database", font_size=28).move_to([-3.4, 3.0, 0])
        self.at("millions")
        self.play(FadeIn(db), LaggedStart(*[FadeIn(d) for d in cloud], lag_ratio=0.003), run_time=1.0)
        q = Star(n=5, outer_radius=0.16, color=YELLOW, fill_opacity=1).move_to([-3.3, -0.1, 0])
        near = sorted(cloud, key=lambda d: np.linalg.norm(d.get_center() - q.get_center()))[:6]
        self.at("nearest")
        self.play(FadeIn(q, scale=2), *[d.animate.set_color(YELLOW).scale(1.8) for d in near], run_time=0.7)
        approx = Text("approximate, but fast", font_size=22, color=GREY_A).next_to(db, DOWN, buff=0.2)
        self.at("approximately")
        self.play(FadeIn(approx), run_time=0.4)
        cards = VGroup(note("close in meaning ≠ correct", RED_E, 24),
                       note("names · codes · numbers → keyword search", BLUE_E, 24),
                       note("hybrid: use both", GREEN_E, 24)).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        cards.move_to([3.3, 0, 0])
        for card, cue in zip(cards, ["close", "exact", "combine"]):
            self.at(cue)
            self.play(FadeIn(card, shift=0.2 * LEFT), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
