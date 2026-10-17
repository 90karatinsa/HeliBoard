// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.keyboard

import kotlin.test.Test
import kotlin.test.assertEquals

/** Tests of the pure decision rule, with three keys of width 100 and height 130 in a row: A (0..100), B (100..200), C (200..300). */
class AdaptiveKeyHitTestTest {
    private val left = intArrayOf(0, 100, 200)
    private val top = intArrayOf(0, 0, 0)
    private val width = intArrayOf(100, 100, 100)
    private val height = intArrayOf(130, 130, 130)

    private fun decide(x: Float, probability: FloatArray, lambda: Float = 1f, y: Float = 65f): Int {
        val geometric = when {
            x < 100f -> 0
            x < 200f -> 1
            else -> 2
        }
        return AdaptiveKeyHitTest.decide(x, y, left, top, width, height, probability, 3, geometric, lambda)
    }

    @Test fun `geometry wins when probabilities are equal`() {
        val equal = floatArrayOf(0.3f, 0.3f, 0.3f)
        assertEquals(0, decide(95f, equal))
        assertEquals(1, decide(105f, equal))
        assertEquals(1, decide(195f, equal))
        assertEquals(2, decide(205f, equal))
    }

    @Test fun `likely neighbour claims touches near the boundary`() {
        // B is very likely, A and C are not
        val bLikely = floatArrayOf(0.02f, 0.9f, 0.02f)
        assertEquals(1, decide(90f, bLikely), "touch 10 px inside A near B goes to B")
        assertEquals(1, decide(210f, bLikely), "touch 10 px inside C near B goes to B")
        // the reverse: A likely, B not
        val aLikely = floatArrayOf(0.9f, 0.02f, 0.05f)
        assertEquals(0, decide(110f, aLikely), "touch 10 px inside B near A goes to A")
    }

    @Test fun `core of the geometric key is never taken away`() {
        val bLikely = floatArrayOf(0.001f, 0.99f, 0.001f)
        assertEquals(0, decide(50f, bLikely), "centre of A stays A")
        assertEquals(0, decide(72f, bLikely), "inside the 60 % core of A stays A")
        assertEquals(2, decide(228f, bLikely), "inside the 60 % core of C stays C")
    }

    @Test fun `reach is capped even for extreme probability ratios`() {
        val bLikely = floatArrayOf(0.0001f, 0.9998f, 0.0001f)
        // a touch in a gap (no geometric key): 0.5 + REACH_X = 0.95 key widths from B's centre (150) -> B can claim x > 55
        assertEquals(1, AdaptiveKeyHitTest.decide(60f, 65f, left, top, width, height, bLikely, 3, -1, 2f))
        assertEquals(0, AdaptiveKeyHitTest.decide(54f, 65f, left, top, width, height, bLikely, 3, -1, 2f))
        // with a geometric key, its core (60 % around the centre, x 20..80 for A) protects it long before the reach cap
        assertEquals(1, decide(85f, bLikely, lambda = 2f))
        assertEquals(0, decide(78f, bLikely, lambda = 2f))
    }

    @Test fun `boundary shift grows with lambda and probability ratio but stops at the core of the neighbour`() {
        fun boundary(probabilities: FloatArray, lambda: Float): Int {
            var x = 100
            while (x < 200 && decide(x.toFloat(), probabilities, lambda) == 0) x++
            return x
        }
        val mild = floatArrayOf(0.3f, 0.2f, 0.1f)
        val medium = floatArrayOf(0.5f, 0.1f, 0.1f)
        val strong = floatArrayOf(0.8f, 0.01f, 0.1f)
        assertEquals(100, boundary(mild, 0f), "lambda 0 disables the adaptation")
        val mildShift = boundary(mild, 1f) - 100
        val mediumShift = boundary(medium, 0.5f) - 100
        val strongShift = boundary(strong, 1f) - 100
        val strongerLambdaShift = boundary(strong, 2f) - 100
        assert(mildShift in 2..10) { "mild shift $mildShift" }
        assert(mediumShift > mildShift) { "medium $mediumShift > mild $mildShift" }
        assert(strongShift > mediumShift) { "strong $strongShift > medium $mediumShift" }
        // B's core starts 0.3 widths before its centre, i.e. at x = 120: no ratio and no lambda moves the boundary past it
        val coreLimit = Math.round((1f - AdaptiveKeyHitTest.CORE) / 2f * 100f)
        assertEquals(coreLimit, strongShift, "strong ratio reaches the core limit")
        assertEquals(coreLimit, strongerLambdaShift, "a larger lambda cannot pass the core limit")
    }

    @Test fun `vertical offsets use the key height`() {
        // key B directly below A' (same x range), A' likely
        val l = intArrayOf(0, 0)
        val t = intArrayOf(0, 130)
        val w = intArrayOf(100, 100)
        val h = intArrayOf(130, 130)
        val upperLikely = floatArrayOf(0.9f, 0.02f)
        assertEquals(0, AdaptiveKeyHitTest.decide(50f, 140f, l, t, w, h, upperLikely, 2, 1, 1f), "just below the boundary goes up")
        assertEquals(1, AdaptiveKeyHitTest.decide(50f, 195f, l, t, w, h, upperLikely, 2, 1, 1f), "centre of the lower key stays")
    }

    @Test fun `only a more likely neighbour takes a touch, also at corners of staggered rows and over padding`() {
        // P in the upper row (150..250) sits above the right half of L in the lower row (100..200, 130..260); the
        // padding right of L (200..250) is part of L's hitbox but not of its drawn rectangle.
        val left2 = intArrayOf(150, 100)
        val top2 = intArrayOf(0, 130)
        val width2 = intArrayOf(100, 100)
        val height2 = intArrayOf(130, 130)
        fun decide2(x: Float, y: Float, p: Float, l: Float) =
            AdaptiveKeyHitTest.decide(x, y, left2, top2, width2, height2, floatArrayOf(p, l), 2, 1, 1f)
        // top right corner of L: P's centre is nearer than L's
        assertEquals(1, decide2(195f, 135f, 0.3f, 0.3f), "equal probabilities: the key under the finger")
        assertEquals(1, decide2(195f, 135f, 0.24f, 0.3f), "a less likely neighbour never takes the touch")
        assertEquals(0, decide2(195f, 135f, 0.45f, 0.3f), "a more likely neighbour does")
        // over the padding, where L's drawn rectangle ends but its hitbox goes on
        assertEquals(1, decide2(240f, 135f, 0.3f, 0.3f))
        assertEquals(1, decide2(240f, 135f, 0.24f, 0.3f))
        assertEquals(0, decide2(240f, 135f, 0.45f, 0.3f))
    }

    @Test fun `no geometric key and no candidates are handled`() {
        assertEquals(-1, AdaptiveKeyHitTest.decide(50f, 65f, left, top, width, height, floatArrayOf(0.3f, 0.3f, 0.3f), 0, -1, 1f))
        // no geometric key (touch in a gap): the most plausible nearby key is chosen
        assertEquals(1, AdaptiveKeyHitTest.decide(150f, 65f, left, top, width, height, floatArrayOf(0.3f, 0.3f, 0.3f), 3, -1, 1f))
    }
}
