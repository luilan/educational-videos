package io.github.luilan.stepstone

import io.github.luilan.stepstone.data.Grading
import io.github.luilan.stepstone.data.Key
import io.github.luilan.stepstone.data.NumberPart
import io.github.luilan.stepstone.data.Parse
import org.junit.Assert.*
import org.junit.Test

class GradingTest {
    private val p028 = NumberPart(null, 0.28, 0.005, null)

    @Test fun parsesNumbers() {
        assertEquals(1234.5, Grading.parseNumber("1,234.5")!!.value, 1e-12)
        assertEquals(0.28, Grading.parseNumber("0,28")!!.value, 1e-12)        // decimal comma
        assertEquals(-3.3, Grading.parseNumber("−3.3")!!.value, 1e-12)        // unicode minus
        assertTrue(Grading.parseNumber("55 %")!!.percent)
        assertEquals(9444096.0, Grading.parseNumber("9,444,096")!!.value, 1e-9)
        assertNull(Grading.parseNumber("abc"))
        assertNull(Grading.parseNumber(""))
    }

    @Test fun numberTolerance() {
        assertTrue(Grading.numberPartRight(p028, "0.28"))
        assertTrue(Grading.numberPartRight(p028, "0.2849"))
        assertFalse(Grading.numberPartRight(p028, "0.29"))
        assertTrue(Grading.numberPartRight(NumberPart(null, 12.0, 0.5, null), "12"))
    }

    @Test fun percentAcceptsFraction() {
        val p = NumberPart(null, 74.0, 1.48, "%")
        assertTrue(Grading.numberPartRight(p, "74"))
        assertTrue(Grading.numberPartRight(p, "74%"))
        assertTrue(Grading.numberPartRight(p, "0.74"))
        assertFalse(Grading.numberPartRight(p, "0.70"))
        // a plain part also accepts the same value written as a percentage
        assertTrue(Grading.numberPartRight(p028, "28%"))
    }

    @Test fun multiPartPartialCredit() {
        val k = Key.Numbers(listOf(NumberPart("(a)", 4.0, 0.5, null), NumberPart("(b)", 48.0, 0.5, null)))
        assertEquals(1.0, Grading.numberScore(k, listOf("4", "48")), 1e-12)
        assertEquals(0.5, Grading.numberScore(k, listOf("4", "47")), 1e-12)
    }

    @Test fun gate() {
        assertTrue(Grading.passed(listOf(1.0, 1.0, 1.0, 0.0), 4))   // 3 of 4
        assertFalse(Grading.passed(listOf(1.0, 1.0, 0.0, 0.0), 4))  // 2 of 4
        assertTrue(Grading.passed(listOf(1.0, 1.0, 0.5), 3))        // 2.5 of 3 >= 2.25
        assertFalse(Grading.passed(listOf(1.0, 1.0, 0.0), 3))
    }

    @Test fun orderAndChoice() {
        assertEquals(1.0, Grading.orderScore(Key.Order(listOf("a", "b")), listOf("a", "b")), 0.0)
        assertEquals(0.0, Grading.orderScore(Key.Order(listOf("a", "b")), listOf("b", "a")), 0.0)
        assertEquals(1.0, Grading.choiceScore(Key.Choice(1), 1), 0.0)
        assertEquals(0.0, Grading.trueFalseScore(Key.TrueFalse(true), null), 0.0)
    }

    @Test fun parsesLessonJson() {
        val json = """{"format":1,"id":"v05","number":5,"title":"T","video":{"youtube":"abc","offline":null},
          "concepts":[{"id":"c1","title":"C","segment":[8,43],"blocks":[{"type":"text","html":"<b>x</b>"},{"type":"future"}],
          "exercises":[{"id":"c1e1","kind":"number","prompt":[],"grading":"auto",
            "key":{"parts":[{"label":null,"value":0.28,"tol":0.005,"unit":null}]},"answer":[],"why":[]},
           {"id":"c1e2","kind":"short","prompt":[],"grading":"self","answer":[],"why":[]}]}]}"""
        val l = Parse.lesson(json)
        assertEquals("abc", l.video.youtube)
        assertEquals(8.0, l.concepts[0].segment.first, 0.0)
        assertTrue(l.concepts[0].blocks[1] is io.github.luilan.stepstone.data.Block.Unknown)
        assertTrue(l.concepts[0].exercises[0].key is Key.Numbers)
        assertTrue(l.concepts[0].exercises[1].selfGraded)
    }
}
