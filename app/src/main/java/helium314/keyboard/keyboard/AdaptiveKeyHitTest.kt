// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.keyboard

import helium314.keyboard.accessibility.AccessibilityUtils
import helium314.keyboard.latin.define.DebugFlags
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.settings.SettingsValues
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.NextCharPredictor
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max

/**
 * Adaptive, invisible touch areas, as introduced by the original iPhone keyboard: the drawn keys never change, but the
 * area that a touch is attributed to grows for letters that are likely to come next in the word being typed and
 * shrinks for unlikely letters.
 *
 * Implemented as a Bayesian decision among the letter keys near the touch: each candidate key is scored with the
 * log-likelihood of the touch under a Gaussian touch model centred on the key, plus the log-probability of its
 * letter being typed next (from [NextCharPredictor]), weighted by the user's intensity setting. The key found by plain
 * geometry keeps the touch unless another key scores clearly better, the touch is in the geometric key's core, or the
 * other key would have to reach too far beyond its own edge.
 *
 * The boundary between two neighbouring keys moves by about σ²·λ·ln(P₁/P₂) key widths, i.e. roughly a quarter of a
 * key for a 10:1 probability ratio at the default intensity, bounded by the reach limits below.
 */
object AdaptiveKeyHitTest {
    private const val TAG = "AdaptiveKeyHitTest"

    /** Standard deviation of the touch offset from the key centre, in key widths / heights. */
    const val SIGMA = 0.35f
    /** Touches closer than this fraction of the half width / height to the geometric key's centre are never moved. */
    const val CORE = 0.6f
    /** A key can claim touches at most this far beyond its own hitbox edge, in key widths. */
    const val REACH_X = 0.45f
    /** A key can claim touches at most this far beyond its own hitbox edge, in key heights. */
    const val REACH_Y = 0.40f
    /** Minimum score advantage over the geometric key needed to move a touch. */
    const val MARGIN = 0.05f
    /** Probabilities are floored here so that unlikely keys shrink but never vanish. */
    const val PROBABILITY_FLOOR = 1e-3f
    /** Language weight λ = this factor × the intensity setting (0..1). */
    const val LAMBDA_PER_INTENSITY = 2f

    /**
     * Pure decision rule (no Android types, so it can be tested and simulated).
     *
     * Candidates are given as parallel arrays of their rectangles (left, top, width, height) and the probability that
     * their letter is typed next. Returns the index of the candidate that should receive the touch at ([x], [y]), which
     * is [geometricIndex] (the candidate chosen by plain geometry, or -1 if none) unless another candidate wins.
     * [lambda] is the weight of the language model term; 0 disables the adaptation.
     */
    @JvmStatic
    fun decide(
        x: Float, y: Float,
        left: IntArray, top: IntArray, width: IntArray, height: IntArray, probability: FloatArray, count: Int,
        geometricIndex: Int, lambda: Float,
    ): Int {
        if (lambda <= 0f || count == 0) return geometricIndex
        var best = geometricIndex
        var bestScore = Float.NEGATIVE_INFINITY
        var geometricProbability = Float.NEGATIVE_INFINITY
        if (geometricIndex in 0 until count) {
            // The hitbox of a key can extend beyond its drawn rectangle (over the gaps, and over the padding at the
            // end of a row); a touch there counts as a touch at the nearest edge of the key.
            val gx = x.coerceIn(left[geometricIndex].toFloat(), (left[geometricIndex] + width[geometricIndex]).toFloat())
            val gy = y.coerceIn(top[geometricIndex].toFloat(), (top[geometricIndex] + height[geometricIndex]).toFloat())
            val dx = normalizedOffset(gx, left[geometricIndex], width[geometricIndex])
            val dy = normalizedOffset(gy, top[geometricIndex], height[geometricIndex])
            if (abs(dx) <= CORE * 0.5f && abs(dy) <= CORE * 0.5f) return geometricIndex
            bestScore = score(dx, dy, probability[geometricIndex], lambda)
            geometricProbability = probability[geometricIndex]
        }
        for (i in 0 until count) {
            if (i == geometricIndex) continue
            // Only a more likely letter may take the touch: without evidence from the language model the key the
            // user touched wins, even where a neighbour's centre happens to be nearer (corners of staggered rows).
            if (probability[i] <= geometricProbability) continue
            val dx = normalizedOffset(x, left[i], width[i])
            val dy = normalizedOffset(y, top[i], height[i])
            if (abs(dx) > 0.5f + REACH_X || abs(dy) > 0.5f + REACH_Y) continue
            val candidateScore = score(dx, dy, probability[i], lambda)
            if (candidateScore > bestScore + MARGIN) {
                bestScore = candidateScore
                best = i
            }
        }
        return best
    }

    /** Offset of [position] from the centre of a hitbox, in units of its size. */
    private fun normalizedOffset(position: Float, start: Int, size: Int): Float {
        val extent = max(1, size).toFloat()
        return (position - (start + extent * 0.5f)) / extent
    }

    /** Log-posterior score of a candidate, higher is better. */
    private fun score(dx: Float, dy: Float, probability: Float, lambda: Float): Float =
        -(dx * dx + dy * dy) / (2f * SIGMA * SIGMA) + lambda * ln(max(probability, PROBABILITY_FLOOR))

    /**
     * Applies the adaptive touch areas to a detected key: returns the key that should receive the touch at
     * ([x], [y]) given [geometricKey] (the key found by plain geometry) and the keys near the touch.
     * [pendingKeyCodes] are the codes of keys held down by older touches, see [NextCharPredictor.currentDistribution].
     */
    @JvmStatic
    fun adjust(keyboard: Keyboard, x: Int, y: Int, geometricKey: Key?, nearestKeys: List<Key>, pendingKeyCodes: IntArray?): Key? {
        if (geometricKey == null || !isLetterKey(geometricKey)) return geometricKey
        val settings: SettingsValues = Settings.getValues() ?: return geometricKey
        if (!settings.mAdaptiveTouchAreas || settings.mAdaptiveTouchAreaIntensity <= 0f) return geometricKey
        if (!keyboard.mId.element.isAlphabet) return geometricKey
        if (!NextCharPredictor.isSupportedLocale(keyboard.mId.locale)) return geometricKey
        if (!settings.needsToLookupSuggestions()) return geometricKey
        if (AccessibilityUtils.instance.isTouchExplorationEnabled) return geometricKey
        val distribution = NextCharPredictor.currentDistribution(keyboard.mId.locale, pendingKeyCodes) ?: return geometricKey

        val capacity = nearestKeys.size + 1
        val candidates = arrayOfNulls<Key>(capacity)
        val left = IntArray(capacity)
        val top = IntArray(capacity)
        val width = IntArray(capacity)
        val height = IntArray(capacity)
        val probability = FloatArray(capacity)
        var count = 0
        var geometricIndex = -1
        for (key in nearestKeys) {
            if (!isLetterKey(key)) continue
            if (key === geometricKey) geometricIndex = count
            count = addCandidate(key, distribution, candidates, left, top, width, height, probability, count)
        }
        if (geometricIndex < 0) {
            geometricIndex = count
            count = addCandidate(geometricKey, distribution, candidates, left, top, width, height, probability, count)
        }
        val lambda = LAMBDA_PER_INTENSITY * settings.mAdaptiveTouchAreaIntensity
        val chosen = decide(x.toFloat(), y.toFloat(), left, top, width, height, probability, count, geometricIndex, lambda)
        val result = candidates[chosen] ?: geometricKey
        if (DebugFlags.DEBUG_ENABLED && result !== geometricKey) {
            Log.d(TAG, "touch at $x,$y moved from '${geometricKey.label}' (p=${probability[geometricIndex]}) to '${result.label}' (p=${probability[chosen]})"
                    + " for prefix '${distribution.prefix}', dictionary evidence: ${distribution.hasDictionaryEvidence}")
        }
        return result
    }

    private fun addCandidate(
        key: Key, distribution: NextCharPredictor.Distribution, candidates: Array<Key?>,
        left: IntArray, top: IntArray, width: IntArray, height: IntArray, probability: FloatArray, count: Int,
    ): Int {
        // the drawn rectangle, not the hitbox: hitboxes of keys at the edge of a row extend over the padding, which
        // would move the centre and the core zone of such keys away from where the user sees them
        candidates[count] = key
        left[count] = key.x
        top[count] = key.y
        width[count] = key.width
        height[count] = key.height
        probability[count] = distribution.probabilityOf(key.code)
        return count + 1
    }

    /** Whether [key] types a single letter and may therefore take part in the adaptation. */
    @JvmStatic
    fun isLetterKey(key: Key): Boolean {
        val code = key.code
        return code > 0 && Character.isLetter(code) && key.outputText == null && !key.isModifier && key.isEnabled
    }
}
