// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import helium314.keyboard.latin.WordComposer
import helium314.keyboard.latin.utils.ScriptUtils.script
import java.util.Locale

/**
 * Probability distribution of the next character of the word currently being typed, used by the adaptive touch areas
 * ([helium314.keyboard.keyboard.AdaptiveKeyHitTest]).
 *
 * Two sources are combined:
 *  - Dictionary completions: after every suggestion computation, [helium314.keyboard.latin.Suggest] publishes the
 *    next-character weights derived from the suggestions that continue the typed prefix (or, for an empty prefix,
 *    the first letters of the predicted next words). Suggestions are computed with a delay and may belong to an older
 *    prefix, so every publication is tagged with the prefix and a generation counter and is only used while both match.
 *  - The character n-gram model of the language being typed ([CharNgramModels]), which is always available
 *    synchronously once built and also covers words that are in no dictionary.
 *
 * The current composing state is read live from the input logic through [InputState], so the result is never stale.
 * Everything here runs on the UI thread except [publish], which runs on the suggestion thread.
 */
object NextCharPredictor {
    /** Weight of a suggestion decays by this factor per rank when deriving next-character weights. */
    const val RANK_DECAY = 0.85f
    /** Maximum weight of the dictionary completion source when mixed with the character model. */
    private const val MAX_DICTIONARY_WEIGHT = 0.85f
    /** Number of contributing suggestions at which the dictionary source reaches half of its maximum weight. */
    private const val DICTIONARY_HALF_EVIDENCE = 4f

    /**
     * Scripts in which one key press types one letter of a word, so that a next-letter model applies. Scripts with
     * combining input (Hangul), abugidas with vowel signs, and scripts without word spaces are left alone.
     */
    private val SUPPORTED_SCRIPTS = setOf(
        ScriptUtils.SCRIPT_LATIN, ScriptUtils.SCRIPT_CYRILLIC, ScriptUtils.SCRIPT_GREEK, ScriptUtils.SCRIPT_ARMENIAN,
        ScriptUtils.SCRIPT_GEORGIAN, ScriptUtils.SCRIPT_HEBREW, ScriptUtils.SCRIPT_ARABIC,
    )

    /** The live typing state, provided by the input logic. Read on the UI thread only. */
    interface InputState {
        /** The composer holding the word being typed. */
        val wordComposer: WordComposer?
        /**
         * The locale of the language currently being typed: with multilingual typing the one the dictionaries judge
         * most likely, which may differ from the keyboard's locale. Null or an empty locale if unknown.
         */
        val locale: Locale?
        /** Whether the cursor is at the start of a word (start of the text or after a word separator) while no word is composed. */
        val isCursorAtWordStart: Boolean
        /** Whether typing [code] ends the current word (space, punctuation, ...). */
        fun isWordSeparator(code: Int): Boolean
    }

    @Volatile
    private var inputState: InputState? = null

    @Volatile
    private var generation = 0

    @Volatile
    private var published: Published? = null

    internal class Published(val prefix: String, val generation: Int, val codePoints: IntArray, val probabilities: FloatArray, val evidence: Int)

    // cache of the last computed distribution, UI thread only
    private var cachedPrefix: String? = null
    private var cachedGeneration = -1
    private var cachedPublished: Published? = null
    private var cachedModel: CharNgramModel? = null
    private var cachedLocale: Locale? = null
    private var cachedDistribution: Distribution? = null

    /** Registers the typing state; called once by the input logic. */
    @JvmStatic
    fun setInputState(state: InputState?) {
        inputState = state
        cachedDistribution = null
        cachedPrefix = null
    }

    /** Whether next-character prediction is meaningful for the script of [locale]. */
    @JvmStatic
    fun isSupportedLocale(locale: Locale): Boolean = locale.script() in SUPPORTED_SCRIPTS

    /**
     * The current generation. The input logic reads it on the UI thread when it requests suggestions and passes it
     * through to [publish], so that suggestions started before a commit are never applied after it.
     */
    @JvmStatic
    fun getGeneration(): Int = generation

    /**
     * Must be called whenever a word is committed or the composing state is reset, so that a distribution
     * published before is never applied to the next word.
     */
    @JvmStatic
    fun onComposingReset() {
        generation++
        published = null
    }

    /**
     * Publishes next-character [weights] (lower-case code point -> weight; only letters are used) that were derived
     * for [prefix] from [evidence] suggestions requested while the generation was [generation].
     */
    @JvmStatic
    fun publish(prefix: String, generation: Int, weights: Map<Int, Float>, evidence: Int, locale: Locale) {
        val folded = HashMap<Int, Float>(weights.size * 2)
        for ((codePoint, weight) in weights) {
            if (weight <= 0f || !Character.isLetter(codePoint)) continue
            val lower = foldToLowerCase(codePoint, locale)
            folded[lower] = (folded[lower] ?: 0f) + weight
        }
        if (folded.isEmpty() || evidence <= 0) {
            published = null
            return
        }
        val codePoints = folded.keys.toIntArray()
        codePoints.sort()
        val probabilities = FloatArray(codePoints.size)
        var sum = 0f
        for (i in codePoints.indices) {
            probabilities[i] = folded[codePoints[i]]!!
            sum += probabilities[i]
        }
        for (i in probabilities.indices) probabilities[i] /= sum
        published = Published(prefix, generation, codePoints, probabilities, evidence)
    }

    /** Forgets the published dictionary distribution, e.g. when suggestions for the prefix yielded nothing usable. */
    @JvmStatic
    fun clearPublished() {
        published = null
    }

    /**
     * The distribution for the current composing state, or null when nothing is known or the state is not one where
     * the next letter can be predicted: no input state, gesture input, combining input, cursor inside a word that is
     * not being composed, no model and no usable suggestions. UI thread only.
     *
     * [keyboardLocale] is used when the input state does not know the language being typed. [pendingKeyCodes] are
     * the codes of keys still held down by older touches, which will be typed before the key of this touch (rollover
     * typing with two thumbs); letters extend the prefix, a word separator starts a new word, anything else gives null.
     */
    @JvmStatic
    @JvmOverloads
    fun currentDistribution(keyboardLocale: Locale, pendingKeyCodes: IntArray? = null): Distribution? {
        val state = inputState ?: return null
        val composer = state.wordComposer ?: return null
        if (composer.isBatchMode) return null
        // input that combines several key presses into one character (e.g. Hangul) is not modelled per key
        if (!composer.combiningSpec.isNullOrEmpty()) return null
        // after tapping into a word the first letter is typed without composing: that is not a word start
        if (!composer.isComposingWord && !state.isCursorAtWordStart) return null
        val locale = state.locale?.takeIf { it.language.isNotEmpty() } ?: keyboardLocale
        val prefix = effectivePrefix(state, composer, pendingKeyCodes, locale) ?: return null
        val currentGeneration = generation
        val current = published
        val validPublished = if (current != null && current.generation == currentGeneration && current.prefix == prefix) current else null
        val model = CharNgramModels.get(locale)
        if (cachedDistribution != null && prefix == cachedPrefix && currentGeneration == cachedGeneration
            && validPublished === cachedPublished && model === cachedModel && locale == cachedLocale) {
            return cachedDistribution
        }
        val distribution = if (validPublished == null && model == null) null
            else Distribution(locale, prefix, model, validPublished)
        cachedPrefix = prefix
        cachedGeneration = currentGeneration
        cachedPublished = validPublished
        cachedModel = model
        cachedLocale = locale
        cachedDistribution = distribution
        return distribution
    }

    /** The typed word up to the cursor, followed by what the keys held down by older touches will type. */
    private fun effectivePrefix(state: InputState, composer: WordComposer, pendingKeyCodes: IntArray?, locale: Locale): String? {
        val typed = currentPrefix(composer)
        if (pendingKeyCodes == null || pendingKeyCodes.isEmpty()) return typed
        val prefix = StringBuilder(typed)
        for (code in pendingKeyCodes) {
            when {
                code > 0 && Character.isLetter(code) -> prefix.appendCodePoint(foldToLowerCase(code, locale))
                code > 0 && state.isWordSeparator(code) -> prefix.setLength(0)
                else -> return null // delete, enter, ...: the word this touch belongs to is unknown
            }
        }
        return prefix.toString()
    }

    /** The typed word up to the cursor. */
    private fun currentPrefix(composer: WordComposer): String {
        val typed = composer.typedWord
        if (typed.isEmpty() || !composer.isCursorFrontOrMiddleOfComposingWord) return typed
        val codePoints = Character.codePointCount(typed, 0, typed.length)
        val position = composer.cursorPositionWithinWord.coerceIn(0, codePoints)
        return typed.substring(0, typed.offsetByCodePoints(0, position))
    }

    /** Lower-cases a code point with the given locale (so Turkish 'I' becomes 'ı'), keeping only the base character. */
    @JvmStatic
    fun foldToLowerCase(codePoint: Int, locale: Locale): Int {
        if (!Character.isUpperCase(codePoint) && !Character.isTitleCase(codePoint)) return codePoint
        val lowered = String(Character.toChars(codePoint)).lowercase(locale)
        return if (lowered.isEmpty()) Character.toLowerCase(codePoint) else lowered.codePointAt(0)
    }

    /** Next-character probabilities for one composing state. Immutable; safe to keep for the duration of a touch. */
    class Distribution internal constructor(
        private val locale: Locale,
        /** The prefix this distribution was computed for. */
        val prefix: String,
        private val model: CharNgramModel?,
        private val dictionary: Published?,
    ) {
        private val lmProbabilities: FloatArray? = model?.nextSymbolProbabilities(prefix)
        private val lmLetterMass: Float
        private val dictionaryWeight: Float
        private val lowerCaseCache = HashMap<Int, Int>()

        init {
            var mass = 0f
            if (lmProbabilities != null) {
                for (s in lmProbabilities.indices) {
                    if (s != CharNgramModel.BOUNDARY) mass += lmProbabilities[s]
                }
            }
            lmLetterMass = if (mass > 0f) mass else 1f
            dictionaryWeight = if (dictionary == null) 0f
                else MAX_DICTIONARY_WEIGHT * dictionary.evidence / (dictionary.evidence + DICTIONARY_HALF_EVIDENCE)
        }

        /** Whether dictionary suggestions contributed (otherwise only the character model did). */
        val hasDictionaryEvidence get() = dictionary != null

        /**
         * Probability, relative to all letters, that the key with [codePoint] (any case) is typed next.
         * 0 if the character is unknown to every source.
         */
        fun probabilityOf(codePoint: Int): Float {
            val folded = lowerCaseCache.getOrPut(codePoint) { foldToLowerCase(codePoint, locale) }
            var dictionaryProbability = -1f
            if (dictionary != null) {
                val index = dictionary.codePoints.binarySearch(folded)
                dictionaryProbability = if (index >= 0) dictionary.probabilities[index] else 0f
            }
            var modelProbability = -1f
            if (model != null && lmProbabilities != null) {
                modelProbability = model.probabilityOfCodePoint(lmProbabilities, folded) / lmLetterMass
            }
            return when {
                dictionaryProbability >= 0f && modelProbability >= 0f ->
                    dictionaryWeight * dictionaryProbability + (1f - dictionaryWeight) * modelProbability
                dictionaryProbability >= 0f -> dictionaryProbability
                modelProbability >= 0f -> modelProbability
                else -> 0f
            }
        }
    }
}
