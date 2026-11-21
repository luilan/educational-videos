"""How AI Agents Work: Inside the Harness, episode 11 — Extending a Harness.

Render from the repo root:  ./render.sh agents a11
Runs: code/a11_extending_a_harness/mcp_harness.py with tickets_mcp.py (2026-11-21, Qwen3 4B on Ollama; logs in
run_2026-11-21.txt). Codex: codex_mcp_capture_2026-11-21.json (Codex CLI 0.163.0, the same server) and openai/codex @
89461ea (core/src/tools/handlers/tool_search_spec.rs).
"""
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from a11_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW AI AGENTS WORK  ·  EPISODE 11"
MODEL_C, HARNESS_C, TOOL_C = PURPLE_B, TEAL_C, GOLD
HANDSHAKE = [("→", '{"jsonrpc":"2.0","id":1,"method":"initialize",...}'),
             ("←", '{"id":1,"result":{"serverInfo":{"name":"tickets"},...}}'),
             ("→", '{"jsonrpc":"2.0","method":"notifications/initialized"}'),
             ("→", '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'),
             ("←", '{"id":2,"result":{"tools":[{"name":"get_ticket",\n   "description":"Get the text of a support ticket by its ID, e.g. T-42.",\n   "inputSchema":{"type":"object","properties":{"id":{"type":"string"}}}}]}}')]
CODEX_TOOL = """{"type": "namespace", "name": "mcp__tickets",
 "tools": [{"type": "function", "name": "get_ticket",
            "description": "Get the text of a support ticket by its ID, e.g. T-42.", ...}]}"""
CODEX_CALL = """{"type": "function_call", "name": "get_ticket", "namespace": "mcp__tickets",
 "arguments": "{\\"id\\":\\"T-42\\"}", "call_id": "call_1z08fgh5"}
{"type": "function_call_output", "call_id": "call_1z08fgh5",
 "output": "Wall time: 0.0104 seconds\\nOutput: Customer reports invoice totals ..."}"""
CODE = """class MCPClient:
    def __init__(self, name, argv):
        self.p = subprocess.Popen(argv, stdin=PIPE, stdout=PIPE, text=True)
        self.request("initialize", {"protocolVersion": "2025-06-18", ...})

    def request(self, method, params=None):
        self.n += 1
        msg = {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}
        self.p.stdin.write(json.dumps(msg) + "\\n")      # one line out
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())["result"]   # one line back"""


def card(title, body, color=GREY_B, fs=17, title_fs=17):
    t = Text(body, font=MONO, font_size=fs, line_spacing=0.85)
    hd = Text(title, font_size=title_fs, color=color)
    g = VGroup(hd, t).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    r = RoundedRectangle(width=g.width + 0.45, height=g.height + 0.35, corner_radius=0.12, color=color, fill_opacity=0.07)
    return VGroup(r, g.move_to(r))


def box(label, color, w=2.6, h=1.0):
    r = RoundedRectangle(width=w, height=h, corner_radius=0.15, color=color, fill_opacity=0.12)
    return VGroup(r, Text(label, font_size=22, color=color).move_to(r))


class ExtendingHarnessVideo(VoicedScene):
    VIDEO = "a11"

    def construct(self):
        play_token_intro(self, TITLE, 11, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.server()      # 2
        self.handshake()   # 3
        self.renamed()     # 4
        self.call()        # 5
        self.runs()        # 6
        self.then()        # 7
        self.codex()       # 8
        self.cost()        # 9
        self.code()        # 10
        self.outro()       # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1
    def hook(self):
        self.section(1)
        h = box("harness", HARNESS_C, w=2.8).move_to([-3.8, 0.6, 0])
        built = VGroup(Text("shell", font=MONO, font_size=20), Text("apply_patch", font=MONO, font_size=20)).arrange(DOWN, buff=0.15).next_to(h, DOWN, buff=0.3)
        self.at("commands")
        self.play(FadeIn(h), FadeIn(built), run_time=0.4)
        need = VGroup(*[Text(t, font_size=22, color=GREY_A) for t in ("ticket tracker", "documentation", "database")]).arrange(DOWN, buff=0.3).move_to([1.2, 0.6, 0])
        self.at("ticket")
        self.play(FadeIn(need, lag_ratio=0.3), run_time=0.8)
        plug = Text("MCP: Model Context Protocol", font_size=30, color=TOOL_C).move_to([0, -2.2, 0])
        self.at("plug")
        self.play(FadeIn(plug), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2
    def server(self):
        self.section(2)
        self.clear_stage()
        h = box("harness", HARNESS_C).move_to([-4.0, 0.8, 0])
        s = box("tickets_mcp.py", TOOL_C, w=3.2).move_to([3.6, 0.8, 0])
        self.at("program")
        self.play(FadeIn(h), FadeIn(s), run_time=0.4)
        a1 = Arrow(h.get_right() + UP * 0.2, s.get_left() + UP * 0.2, buff=0.1, color=GREY_B)
        a2 = Arrow(s.get_left() + DOWN * 0.2, h.get_right() + DOWN * 0.2, buff=0.1, color=GREY_B)
        l1 = Text("stdin: one JSON-RPC message per line", font_size=17).next_to(a1, UP, buff=0.1)
        l2 = Text("stdout", font_size=17).next_to(a2, DOWN, buff=0.1)
        self.at("json", "talk")
        self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(l1), FadeIn(l2), run_time=0.6)
        info = Text("45 lines of Python, standard library only · one tool: get_ticket", font_size=22, color=GREY_A).move_to([0, -1.5, 0])
        self.at("forty", "45")
        self.play(FadeIn(info), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3
    def handshake(self):
        self.section(3)
        self.clear_stage()
        rows = VGroup(*[VGroup(Text(d, font_size=22, color=HARNESS_C if d == "→" else TOOL_C), Text(m, font=MONO, font_size=16, line_spacing=0.85))
                        .arrange(RIGHT, buff=0.3, aligned_edge=UP) for d, m in HANDSHAKE]).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        rows.move_to([0, 0.2, 0])
        if rows.width > 13:
            rows.scale_to_fit_width(13)
        self.at("handshake")
        self.play(FadeIn(rows[0]), FadeIn(rows[1]), run_time=0.4)
        self.at("then")
        self.play(FadeIn(rows[2]), run_time=0.3)
        self.at("list")
        self.play(FadeIn(rows[3]), run_time=0.3)
        self.at("name")
        self.play(FadeIn(rows[4]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4
    def renamed(self):
        self.section(4)
        self.clear_stage()
        h = Text("tools offered to the model", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        a = Text("shell", font=MONO, font_size=28)
        b = Text("mcp__tickets__get_ticket", font=MONO, font_size=28, color=TOOL_C)
        g = VGroup(a, b).arrange(DOWN, buff=0.5).move_to([0, 0.3, 0])
        self.play(FadeIn(a), run_time=0.3)
        self.at("becomes")
        self.play(FadeIn(b), run_time=0.4)
        same = Text("same shape: name, description, parameters (225 characters each)", font_size=21, color=GREY_A).move_to([0, -1.8, 0])
        self.at("same")
        self.play(FadeIn(same), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5
    def call(self):
        self.section(5)
        self.clear_stage()
        m = box("model", MODEL_C).move_to([-5.0, 0.5, 0])
        h = box("harness", HARNESS_C).move_to([0, 0.5, 0])
        s = box("tickets", TOOL_C).move_to([5.0, 0.5, 0])
        self.play(FadeIn(m), FadeIn(h), FadeIn(s), run_time=0.4)
        c = Text('mcp__tickets__get_ticket({"id": "T-42"})', font=MONO, font_size=16).move_to([-2.5, 1.5, 0])
        a1 = Arrow(m.get_right(), h.get_left(), buff=0.1, color=GREY_B)
        self.at("calls")
        self.play(GrowArrow(a1), FadeIn(c), run_time=0.4)
        f = Text('tools/call {"name": "get_ticket", ...}', font=MONO, font_size=16).move_to([2.5, -0.35, 0])
        a2 = Arrow(h.get_right(), s.get_left(), buff=0.1, color=GREY_B)
        self.at("forwards")
        self.play(GrowArrow(a2), FadeIn(f), run_time=0.4)
        r = Text("“Customer reports invoice totals are too high by exactly the discount amount. …”", font_size=18, color=GREEN_B).move_to([0, -1.2, 0])
        self.at("back")
        self.play(FadeIn(r), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6
    def runs(self):
        self.section(6)
        self.clear_stage()
        h = Text("task: “Ticket T-42 describes a bug in this project. Read the ticket, fix the bug, …”", font_size=20).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        rows = VGroup(Text("run 1, step 1:  MCP tickets.get_ticket({'id': 'T-42'})", font=MONO, font_size=20, color=TOOL_C),
                      Text("run 2, step 1:  MCP tickets.get_ticket({'id': 'T-42'})", font=MONO, font_size=20, color=TOOL_C)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        rows.move_to([0, 0.6, 0])
        self.at("first")
        self.play(FadeIn(rows), run_time=0.5)
        t = card("the ticket", "Customer reports invoice totals are too high by exactly the discount\namount. Seen on invoice 2026-0042 "
                 "(loyalty discount 10%).\nExpected: total = subtotal - discount + VAT.", color=GREEN_B, fs=18).move_to([0, -1.6, 0])
        self.at("ticket")
        self.play(FadeIn(t), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7
    def then(self):
        self.section(7)
        self.clear_stage()
        rows = VGroup(Text("$ python3 test_invoice.py   → exit 1", font=MONO, font_size=20),
                      Text("run 1: $ sed -i 's/total = subtotal * 1.10 - discount/…/' …", font=MONO, font_size=20),
                      Text("run 2: $ sed -i 's/total = subtotal + discount + vat/…/' …", font=MONO, font_size=20)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        rows.move_to([0, 1.0, 0])
        self.at("test")
        self.play(FadeIn(rows[0]), run_time=0.3)
        self.at("fix")
        self.play(FadeIn(rows[1:]), run_time=0.4)
        n = Text("neither line exists: invoice.py was never read", font_size=22, color=RED_B).move_to([0, -0.6, 0])
        self.play(FadeIn(n), run_time=0.3)
        r = Text("both runs: tests still failing", font_size=24, color=RED_B).move_to([0, -1.4, 0])
        self.at("failing")
        self.play(FadeIn(r), run_time=0.3)
        l = Text("a new tool adds what the model can do, not how well", font_size=24, color=YELLOW).to_edge(DOWN, buff=0.5)
        self.at("adds")
        self.play(FadeIn(l), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8
    def codex(self):
        self.section(8)
        self.clear_stage()
        cfg = card("Codex: two lines of configuration", '-c mcp_servers.tickets.command="python3"\n-c mcp_servers.tickets.args=["/work/tickets_mcp.py"]',
                   color=HARNESS_C, fs=18).to_edge(UP, buff=0.4)
        self.at("configuration")
        self.play(FadeIn(cfg), run_time=0.4)
        t = card("in the request's tool list", CODEX_TOOL, color=TOOL_C, fs=15).move_to([0, 0.4, 0])
        if t.width > 13:
            t.scale_to_fit_width(13)
        self.at("namespace")
        self.play(FadeIn(t), run_time=0.4)
        c = card("the call and its result", CODEX_CALL, color=GREEN_B, fs=15).move_to([0, -2.1, 0])
        if c.width > 13:
            c.scale_to_fit_width(13)
        self.at("called")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9
    def cost(self):
        self.section(9)
        self.clear_stage()
        h = Text("every tool definition is sent on every call", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        grid = VGroup(*[Square(0.32, stroke_width=0, fill_color=TOOL_C, fill_opacity=0.8) for _ in range(100)]).arrange_in_grid(10, 10, buff=0.06)
        grid.move_to([-3.8, -0.3, 0])
        gl = Text("10 servers × 10 tools", font_size=20).next_to(grid, DOWN, buff=0.2)
        self.at("10", "ten")
        self.play(FadeIn(grid, lag_ratio=0.01), FadeIn(gl), run_time=0.8)
        ts = card("Codex: deferred tools", 'tool_search(query: "Search query for deferred tools.",\n            limit)\n→ only the matching tools are loaded',
                  color=HARNESS_C, fs=15)
        ts.move_to([3.4, -0.3, 0])
        if ts.get_right()[0] > 6.9:
            ts.shift(LEFT * (ts.get_right()[0] - 6.9))
        self.at("defer")
        self.play(FadeIn(ts), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=19)
        code.move_to([0, 0.1, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("process")
        self.play(Create(hl), run_time=0.3)
        self.at("write")
        self.play(highlight(hl, code, 8), run_time=0.3)
        self.at("read")
        self.play(highlight(hl, code, 10), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 11
    def outro(self):
        self.section(11)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
