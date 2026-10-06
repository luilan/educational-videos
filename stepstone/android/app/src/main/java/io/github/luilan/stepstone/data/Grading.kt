package io.github.luilan.stepstone.data

import kotlin.math.abs

/** Share of a concept's points needed to lay its stone (unlock the next concept). */
const val PASS_SHARE = 0.75

/** Self-grading choices: got it, partly, missed. */
enum class SelfGrade(val score: Double) { GOT_IT(1.0), PARTLY(0.5), MISSED(0.0) }

data class ParsedNumber(val value: Double, val percent: Boolean)

object Grading {
    private val thousands = Regex("""^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$""")

    /** Reads what a learner typed: "1,234.5", "0,28" (decimal comma), "−3.3", "55 %", "1e-3". Null if not a number. */
    fun parseNumber(raw: String): ParsedNumber? {
        var s = raw.trim().replace('−', '-').replace('–', '-').replace(" ", "").replace(" ", "")
        val percent = s.endsWith("%")
        if (percent) s = s.dropLast(1)
        if (s.isEmpty()) return null
        s = when {
            thousands.matches(s) -> s.replace(",", "")
            s.count { it == ',' } == 1 && !s.contains('.') -> s.replace(',', '.')
            else -> s
        }
        val v = s.toDoubleOrNull() ?: return null
        return if (v.isFinite()) ParsedNumber(v, percent) else null
    }

    /** One part of a number answer. Percent parts also accept the fraction (55% == 0.55) and vice versa. */
    fun numberPartRight(part: NumberPart, raw: String): Boolean {
        val n = parseNumber(raw) ?: return false
        val eps = 1e-9 * maxOf(1.0, abs(part.value))
        fun close(x: Double) = abs(x - part.value) <= part.tol + eps
        return if (part.unit == "%") {
            close(n.value) || (!n.percent && close(n.value * 100))
        } else {
            close(n.value) || (n.percent && close(n.value / 100))
        }
    }

    /** Score of a number answer: the share of parts that are right (1.0 when all are). */
    fun numberScore(key: Key.Numbers, inputs: List<String>): Double =
        key.parts.indices.count { i -> numberPartRight(key.parts[i], inputs.getOrElse(i) { "" }) }.toDouble() / key.parts.size

    fun orderScore(key: Key.Order, given: List<String>) = if (given == key.items) 1.0 else 0.0
    fun choiceScore(key: Key.Choice, chosen: Int?) = if (chosen == key.index) 1.0 else 0.0
    fun trueFalseScore(key: Key.TrueFalse, chosen: Boolean?) = if (chosen == key.value) 1.0 else 0.0

    /** Concept gate: passed once the points reach PASS_SHARE of the concept's exercises. */
    fun passed(scores: Collection<Double>, exercises: Int): Boolean =
        exercises == 0 || scores.sum() >= PASS_SHARE * exercises - 1e-9

    /** Points needed to pass a concept with this many exercises (3 of 4, 2 of 3...). */
    fun needed(exercises: Int): Double = Math.ceil(PASS_SHARE * exercises - 1e-9)
}
