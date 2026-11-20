"""Base scene that plays narration sections and cues animations to spoken words."""
import json
import os

from manim import Scene, config


class VoicedScene(Scene):
    VIDEO = None          # e.g. "v01"; narration is read from voice/<VIDEO>/sections.json
    SECTION_GAP = 0.5     # silence after each section

    def setup(self):
        with open(os.path.join("voice", self.VIDEO, "sections.json")) as f:
            self.sections = json.load(f)

    def section(self, i):
        """Start narration section i (1-based)."""
        if i > 1:  # report animations that ran past the previous section's narration
            prev = self.sections[i - 2]
            late = self.renderer.time - (self.sec_start + prev["dur"] + self.SECTION_GAP)
            if late > 0.05:
                print(f"section {i - 1} overran its narration by {late:.2f}s")
        self.sec = self.sections[i - 1]
        self.sec_start = self.renderer.time
        self.cursor = 0
        self.add_sound(self.sec["file"])

    def at(self, *words):
        """Wait until the next occurrence of any of `words` is spoken."""
        spoken = self.sec["words"]
        for j in range(self.cursor, len(spoken)):
            if spoken[j]["w"] in words:
                self.cursor = j + 1
                self.wait_until(self.sec_start + spoken[j]["t"])
                return
        raise ValueError(f"cue {words} not found after word {self.cursor} in {self.sec['file']}")

    def end_section(self):
        self.wait_until(self.sec_start + self.sec["dur"] + self.SECTION_GAP)

    def wait_until(self, t):
        if t - self.renderer.time > 1 / config.frame_rate:
            self.wait(t - self.renderer.time)
