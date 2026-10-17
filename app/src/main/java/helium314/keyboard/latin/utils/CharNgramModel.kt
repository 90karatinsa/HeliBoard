// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.IOException
import java.util.Locale
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Character n-gram language model with interpolated Witten-Bell smoothing.
 *
 * The model is trained from the words of a dictionary (each word weighted by its unigram probability) and from the
 * words the user actually types. It answers "given the characters typed so far in the current word, how likely is each
 * character to come next?" for any prefix, also for words that are in no dictionary (e.g. inflected Turkish forms).
 * It is used as the language prior of the adaptive touch areas and as a backoff source for completions.
 *
 * The alphabet is learned from the words, so the model is language agnostic. Characters are folded to lower case with
 * the locale of the model (Turkish dotted/dotless i are kept distinct). Up to [MAX_SYMBOLS] - 2 distinct characters are
 * tracked; everything beyond that is pooled in a single [OTHER] symbol, which makes the model gracefully useless, not
 * wrong, for scripts with huge alphabets.
 *
 * All public methods are synchronized. A model under construction must not be shared before it is complete.
 */
class CharNgramModel(
    /** BCP-47 tag of the locale the model was built for, used for case folding. */
    val localeTag: String,
    /** Opaque key identifying the dictionaries this model was built from (for cache validation). */
    val dictKey: String,
    /** Number of preceding characters used as context; 3 means a 4-gram model. */
    val maxHistory: Int = DEFAULT_MAX_HISTORY,
) {
    private val locale: Locale = Locale.forLanguageTag(localeTag)

    // ---- alphabet ----
    private val symbolCodePoints = IntArray(MAX_SYMBOLS)
    private val codePointToSymbol = HashMap<Int, Int>()
    private val lowerCaseCache = HashMap<Int, Int>()
    /** Number of symbols in use, including [BOUNDARY] and [OTHER]. */
    var symbolCount = FIRST_LETTER_SYMBOL
        private set
    /** Distinct code points that did not fit into the alphabet (diagnostic). */
    var overflowCodePoints = 0
        private set

    // ---- counts: open addressing hash table, key -> (count, distinct successors for context entries) ----
    // Counts are longs: a frequent dictionary word weighs hundreds of thousands (see [dictionaryWeight]) and a
    // dictionary has millions of character transitions, so the context totals exceed the range of an int.
    private var keys = LongArray(INITIAL_CAPACITY)
    private var counts = LongArray(INITIAL_CAPACITY)
    private var distinct = IntArray(INITIAL_CAPACITY)
    /** Number of occupied table slots (n-gram entries plus context entries). */
    var size = 0
        private set
    /** Total weight of all words added. */
    var totalWeight = 0L
        private set
    /** Number of words added (dictionary words and learned words). */
    var wordCount = 0
        private set
    /** Size of the count table in bytes. */
    val memoryBytes: Long
        get() = keys.size.toLong() * (2 * Long.SIZE_BYTES + Int.SIZE_BYTES)

    init {
        require(maxHistory in 1..MAX_HISTORY_LIMIT) { "maxHistory must be in 1..$MAX_HISTORY_LIMIT" }
        symbolCodePoints[BOUNDARY] = -1
        symbolCodePoints[OTHER] = -2
    }

    // ------------------------------------------------------------------ training

    /**
     * Adds [word] with the given [weight] (a positive integer, see [dictionaryWeight] and [USER_WORD_WEIGHT]).
     * Words with characters that are not letters, apostrophes or hyphens are ignored, as are overlong words.
     */
    @Synchronized
    fun addWord(word: CharSequence, weight: Int) {
        if (weight <= 0) return
        val symbols = toSymbols(word, true) ?: return
        wordCount++
        totalWeight += weight
        val n = symbols.size
        for (i in 1 until n) {
            val next = symbols[i]
            val maxH = min(maxHistory, i)
            for (h in 0..maxH) {
                val ctx = packHistory(symbols, i, h)
                val slot = findOrInsert(makeKey(h, ctx, next.toLong()))
                val isNew = counts[slot] == 0L
                counts[slot] += weight
                val ctxSlot = findOrInsert(makeKey(h, ctx, CONTEXT_MARKER))
                counts[ctxSlot] += weight
                if (isNew) distinct[ctxSlot]++
            }
        }
    }

    // ------------------------------------------------------------------ querying

    /**
     * Probability of every symbol (indexed by symbol, size [symbolCount]) being the next character after [prefix].
     * [prefix] is the word typed so far (any case); an empty prefix gives the distribution of first characters.
     * The [BOUNDARY] symbol carries the probability that the word ends here. Probabilities sum to 1.
     */
    @Synchronized
    fun nextSymbolProbabilities(prefix: CharSequence): FloatArray {
        val history = toSymbols(prefix, false, withTrailingBoundary = false)
            ?: intArrayOf(BOUNDARY) // prefix contains unknown stuff: fall back to word-start context
        val n = symbolCount
        val probabilities = FloatArray(n)
        // order 0: add-one smoothed unigrams
        val unigramCtx = findSlot(makeKey(0, 0L, CONTEXT_MARKER))
        val unigramTotal = if (unigramCtx >= 0) counts[unigramCtx].toFloat() else 0f
        for (s in 0 until n) {
            val c = countOf(makeKey(0, 0L, s.toLong()))
            probabilities[s] = (c + 1f) / (unigramTotal + n)
        }
        // higher orders: interpolated Witten-Bell
        val available = history.size // history[0] is BOUNDARY, so at least 1 symbol of context is always there
        val maxH = min(maxHistory, available)
        for (h in 1..maxH) {
            val ctx = packHistory(history, available, h)
            val ctxSlot = findSlot(makeKey(h, ctx, CONTEXT_MARKER))
            if (ctxSlot < 0) break // unseen context: longer contexts are unseen too
            val total = counts[ctxSlot].toFloat()
            val types = distinct[ctxSlot].toFloat()
            if (total <= 0f || types <= 0f) break
            val denominator = total + types
            for (s in 0 until n) {
                val c = countOf(makeKey(h, ctx, s.toLong()))
                probabilities[s] = (c + types * probabilities[s]) / denominator
            }
        }
        return probabilities
    }

    /**
     * Probability that [codePoint] is the next character after [prefix], using [nextSymbolProbabilities].
     * Characters outside the alphabet get a share of the [OTHER] mass.
     */
    @Synchronized
    fun nextCodePointProbability(prefix: CharSequence, codePoint: Int): Float {
        val probabilities = nextSymbolProbabilities(prefix)
        return probabilityOfCodePoint(probabilities, codePoint)
    }

    /** Looks up the probability of [codePoint] in a distribution returned by [nextSymbolProbabilities]. */
    @Synchronized
    fun probabilityOfCodePoint(probabilities: FloatArray, codePoint: Int): Float {
        val symbol = symbolOf(codePoint, false)
        // a symbol added (by learning) after the distribution was computed is beyond its end: treat it as unseen
        return if (symbol == OTHER || symbol < 0 || symbol >= probabilities.size) probabilities[OTHER] * OTHER_SHARE
        else probabilities[symbol]
    }

    /** Log-probability of the whole [word] (including its end), a measure of how plausible the word is. */
    @Synchronized
    fun logProbabilityOfWord(word: CharSequence): Double {
        val symbols = toSymbols(word, false) ?: return Double.NEGATIVE_INFINITY
        var logProbability = 0.0
        for (i in 1 until symbols.size) {
            val probabilities = nextSymbolProbabilitiesForSymbols(symbols, i)
            logProbability += Math.log(probabilities[symbols[i]].toDouble())
        }
        return logProbability
    }

    /** Symbol index of [codePoint] after case folding, [OTHER] if unknown, or -1 if it cannot be part of a word. */
    @Synchronized
    fun symbolOf(codePoint: Int): Int = symbolOf(codePoint, false)

    /** Code point of a letter symbol, or -1 for [BOUNDARY] / [OTHER]. */
    @Synchronized
    fun codePointOf(symbol: Int): Int = if (symbol in FIRST_LETTER_SYMBOL until symbolCount) symbolCodePoints[symbol] else -1

    /** Whether the model has learned enough to be useful. */
    val isUseful: Boolean
        @Synchronized get() = symbolCount >= MIN_USEFUL_SYMBOLS && wordCount >= MIN_USEFUL_WORDS
                && overflowCodePoints < MAX_OVERFLOW_CODE_POINTS

    // ------------------------------------------------------------------ completion

    /**
     * Most probable continuations of [prefix] up to a word end, found by beam search over the model. Returns up to
     * [maxResults] full words (prefix included, lower case) with their log probability, best first. Only continuations
     * of at least [minExtraChars] and at most [maxExtraChars] characters are returned.
     */
    @Synchronized
    fun complete(prefix: CharSequence, maxResults: Int, minExtraChars: Int = 1, maxExtraChars: Int = 12, beamWidth: Int = 8): List<Pair<String, Double>> {
        val start = toSymbols(prefix, false, withTrailingBoundary = false) ?: return emptyList()
        if (start.size < 2) return emptyList() // needs at least one real character
        var beam = listOf(start to 0.0)
        val results = ArrayList<Pair<String, Double>>()
        for (step in 1..maxExtraChars) {
            val candidates = ArrayList<Pair<IntArray, Double>>()
            for ((symbols, logProbability) in beam) {
                val probabilities = nextSymbolProbabilitiesForSymbols(symbols, symbols.size)
                for (s in 0 until symbolCount) {
                    val p = probabilities[s]
                    if (p < MIN_BEAM_PROBABILITY) continue
                    val newLogProbability = logProbability + Math.log(p.toDouble())
                    if (s == BOUNDARY) {
                        if (step - 1 >= minExtraChars) results.add(symbolsToString(symbols) to newLogProbability)
                    } else if (s != OTHER) {
                        candidates.add((symbols + s) to newLogProbability)
                    }
                }
            }
            if (candidates.isEmpty()) break
            candidates.sortByDescending { it.second }
            beam = candidates.take(beamWidth)
        }
        results.sortByDescending { it.second }
        return results.take(maxResults)
    }

    // ------------------------------------------------------------------ serialization

    @Synchronized
    @Throws(IOException::class)
    fun writeTo(stream: DataOutputStream) {
        stream.writeInt(FILE_MAGIC)
        stream.writeInt(FILE_VERSION)
        stream.writeUTF(localeTag)
        stream.writeUTF(dictKey)
        stream.writeInt(maxHistory)
        stream.writeInt(symbolCount)
        for (s in 0 until symbolCount) stream.writeInt(symbolCodePoints[s])
        stream.writeInt(overflowCodePoints)
        stream.writeInt(wordCount)
        stream.writeLong(totalWeight)
        stream.writeInt(size)
        for (i in keys.indices) {
            if (keys[i] == 0L) continue
            stream.writeLong(keys[i])
            stream.writeLong(counts[i])
            stream.writeInt(distinct[i])
        }
    }

    // ------------------------------------------------------------------ internals

    private fun nextSymbolProbabilitiesForSymbols(symbols: IntArray, position: Int): FloatArray {
        val n = symbolCount
        val probabilities = FloatArray(n)
        val unigramCtx = findSlot(makeKey(0, 0L, CONTEXT_MARKER))
        val unigramTotal = if (unigramCtx >= 0) counts[unigramCtx].toFloat() else 0f
        for (s in 0 until n) {
            probabilities[s] = (countOf(makeKey(0, 0L, s.toLong())) + 1f) / (unigramTotal + n)
        }
        val maxH = min(maxHistory, position)
        for (h in 1..maxH) {
            val ctx = packHistory(symbols, position, h)
            val ctxSlot = findSlot(makeKey(h, ctx, CONTEXT_MARKER))
            if (ctxSlot < 0) break
            val total = counts[ctxSlot].toFloat()
            val types = distinct[ctxSlot].toFloat()
            if (total <= 0f || types <= 0f) break
            val denominator = total + types
            for (s in 0 until n) {
                probabilities[s] = (countOf(makeKey(h, ctx, s.toLong())) + types * probabilities[s]) / denominator
            }
        }
        return probabilities
    }

    private fun symbolsToString(symbols: IntArray): String {
        val sb = StringBuilder()
        for (s in symbols) {
            if (s >= FIRST_LETTER_SYMBOL) sb.appendCodePoint(symbolCodePoints[s])
        }
        return sb.toString()
    }

    /**
     * Converts a word to [BOUNDARY, symbols..., BOUNDARY] (the trailing boundary is optional).
     * Returns null if the word is empty, too long, or contains characters that cannot be part of a word.
     */
    private fun toSymbols(word: CharSequence, allowNewSymbols: Boolean, withTrailingBoundary: Boolean = true): IntArray? {
        val length = word.length
        if (length > MAX_WORD_LENGTH) return null
        val codePointCount = Character.codePointCount(word, 0, length)
        if (codePointCount == 0 && withTrailingBoundary) return null
        val symbols = IntArray(codePointCount + if (withTrailingBoundary) 2 else 1)
        symbols[0] = BOUNDARY
        var i = 0
        var index = 1
        while (i < length) {
            val codePoint = Character.codePointAt(word, i)
            i += Character.charCount(codePoint)
            val symbol = symbolOf(codePoint, allowNewSymbols)
            if (symbol < 0) return null
            symbols[index++] = symbol
        }
        if (withTrailingBoundary) symbols[index] = BOUNDARY
        return symbols
    }

    private fun symbolOf(codePoint: Int, allowNew: Boolean): Int {
        if (!isWordCodePoint(codePoint)) return -1
        val folded = foldCase(codePoint)
        val known = codePointToSymbol[folded]
        if (known != null) return known
        if (!allowNew) return OTHER
        if (symbolCount >= MAX_SYMBOLS) {
            overflowCodePoints++
            codePointToSymbol[folded] = OTHER
            return OTHER
        }
        val symbol = symbolCount++
        symbolCodePoints[symbol] = folded
        codePointToSymbol[folded] = symbol
        return symbol
    }

    private fun foldCase(codePoint: Int): Int {
        if (!Character.isUpperCase(codePoint) && !Character.isTitleCase(codePoint)) return codePoint
        val cached = lowerCaseCache[codePoint]
        if (cached != null) return cached
        val lowered = String(Character.toChars(codePoint)).lowercase(locale)
        // e.g. "İ".lowercase(Locale.ENGLISH) is "i" + combining dot above: keep the base character only
        val folded = if (lowered.isEmpty()) Character.toLowerCase(codePoint) else lowered.codePointAt(0)
        lowerCaseCache[codePoint] = folded
        return folded
    }

    private fun packHistory(symbols: IntArray, position: Int, h: Int): Long {
        var ctx = 0L
        for (j in position - h until position) {
            ctx = (ctx shl SYMBOL_BITS) or symbols[j].toLong()
        }
        return ctx
    }

    private fun makeKey(h: Int, ctx: Long, next: Long): Long =
        KEY_TAG or (h.toLong() shl HISTORY_SHIFT) or (ctx shl SYMBOL_BITS) or next

    private fun countOf(key: Long): Long {
        val slot = findSlot(key)
        return if (slot < 0) 0L else counts[slot]
    }

    private fun findSlot(key: Long): Int {
        val mask = keys.size - 1
        var slot = mix(key) and mask
        while (true) {
            val k = keys[slot]
            if (k == key) return slot
            if (k == 0L) return -1
            slot = (slot + 1) and mask
        }
    }

    private fun findOrInsert(key: Long): Int {
        if (size * 10 >= keys.size * 6) grow()
        val mask = keys.size - 1
        var slot = mix(key) and mask
        while (true) {
            val k = keys[slot]
            if (k == key) return slot
            if (k == 0L) {
                keys[slot] = key
                size++
                return slot
            }
            slot = (slot + 1) and mask
        }
    }

    private fun grow() {
        val oldKeys = keys
        val oldCounts = counts
        val oldDistinct = distinct
        val newCapacity = oldKeys.size * 2
        keys = LongArray(newCapacity)
        counts = LongArray(newCapacity)
        distinct = IntArray(newCapacity)
        val mask = newCapacity - 1
        for (i in oldKeys.indices) {
            val key = oldKeys[i]
            if (key == 0L) continue
            var slot = mix(key) and mask
            while (keys[slot] != 0L) slot = (slot + 1) and mask
            keys[slot] = key
            counts[slot] = oldCounts[i]
            distinct[slot] = oldDistinct[i]
        }
    }

    private fun insertRaw(key: Long, count: Long, distinctCount: Int) {
        val slot = findOrInsert(key)
        counts[slot] = count
        distinct[slot] = distinctCount
    }

    override fun toString(): String =
        "CharNgramModel($localeTag, history=$maxHistory, symbols=$symbolCount, words=$wordCount, entries=$size, overflow=$overflowCodePoints, ${memoryBytes / 1024} KB)"

    companion object {
        const val DEFAULT_MAX_HISTORY = 3
        const val MAX_HISTORY_LIMIT = 4
        /** Symbols are 7 bits; the all-ones value is [CONTEXT_MARKER], so it is never a symbol. */
        const val MAX_SYMBOLS = 127
        /** Symbol for the start and end of a word. */
        const val BOUNDARY = 0
        /** Symbol pooling all characters that did not fit into the alphabet. */
        const val OTHER = 1
        private const val FIRST_LETTER_SYMBOL = 2
        private const val SYMBOL_BITS = 7
        private const val HISTORY_SHIFT = SYMBOL_BITS * (MAX_HISTORY_LIMIT + 1)
        private const val KEY_TAG = 1L shl 62
        /** "next symbol" value of a context aggregate entry (never a real symbol). */
        private const val CONTEXT_MARKER = (1L shl SYMBOL_BITS) - 1
        private const val INITIAL_CAPACITY = 1 shl 14
        private const val MAX_WORD_LENGTH = 48
        private const val OTHER_SHARE = 0.25f
        private const val MIN_BEAM_PROBABILITY = 0.02f
        private const val MIN_USEFUL_SYMBOLS = 8
        private const val MIN_USEFUL_WORDS = 200
        private const val MAX_OVERFLOW_CODE_POINTS = 200
        private const val FILE_MAGIC = 0x48434C4D // "HCLM"
        private const val FILE_VERSION = 2

        /** Weight of one use of a word by the user: about that of a dictionary word in the top tenth by frequency. */
        const val USER_WORD_WEIGHT = 150
        /** Divisor of the dictionary probability in the exponent of [dictionaryWeight]; smaller means frequent words count more. */
        const val DEFAULT_WEIGHT_BASE = 20.0

        /** Training weight of a dictionary word from its unigram probability (0..255, logarithmic scale). */
        @JvmOverloads
        fun dictionaryWeight(probability: Int, base: Double = DEFAULT_WEIGHT_BASE): Int {
            val p = probability.coerceIn(0, 255)
            return exp(p / base).roundToInt().coerceAtLeast(1)
        }

        /** Whether [codePoint] can be part of a word for this model: letters, marks, apostrophes and hyphens. */
        fun isWordCodePoint(codePoint: Int): Boolean {
            if (Character.isLetter(codePoint)) return true
            return when (Character.getType(codePoint).toByte()) {
                Character.NON_SPACING_MARK, Character.COMBINING_SPACING_MARK -> true
                else -> codePoint == '\''.code || codePoint == '’'.code || codePoint == '-'.code
            }
        }

        private fun mix(key: Long): Int {
            var h = key * -0x61c8864680b583ebL // golden ratio hashing
            h = h xor (h ushr 32)
            return h.toInt()
        }

        /** Reads a model written by [writeTo]. */
        @Throws(IOException::class)
        fun read(stream: DataInputStream): CharNgramModel {
            if (stream.readInt() != FILE_MAGIC) throw IOException("not a char n-gram model file")
            if (stream.readInt() != FILE_VERSION) throw IOException("unsupported char n-gram model version")
            val localeTag = stream.readUTF()
            val dictKey = stream.readUTF()
            val maxHistory = stream.readInt()
            val model = CharNgramModel(localeTag, dictKey, maxHistory)
            val symbolCount = stream.readInt()
            if (symbolCount < FIRST_LETTER_SYMBOL || symbolCount > MAX_SYMBOLS) throw IOException("bad symbol count")
            for (s in 0 until symbolCount) {
                val codePoint = stream.readInt()
                model.symbolCodePoints[s] = codePoint
                if (s >= FIRST_LETTER_SYMBOL) model.codePointToSymbol[codePoint] = s
            }
            model.symbolCount = symbolCount
            model.overflowCodePoints = stream.readInt()
            model.wordCount = stream.readInt()
            model.totalWeight = stream.readLong()
            val entries = stream.readInt()
            if (entries < 0) throw IOException("bad entry count")
            for (i in 0 until entries) {
                model.insertRaw(stream.readLong(), stream.readLong(), stream.readInt())
            }
            return model
        }
    }
}
