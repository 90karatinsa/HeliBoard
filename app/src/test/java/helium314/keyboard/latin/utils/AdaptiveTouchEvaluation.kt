// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import helium314.keyboard.keyboard.AdaptiveKeyHitTest
import org.junit.Assume.assumeTrue
import java.io.File
import java.util.Locale
import java.util.Random
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.test.Test

/**
 * Offline evaluation of the adaptive touch areas and the character model. Not a unit test: it only runs when the
 * environment variable CHARLM_EVAL_DIR points to a directory containing, per language tag `xx`,
 *  - `words_xx.tsv`: one `word<TAB>probability` line per dictionary word (dump of the main dictionary), and
 *  - `text_xx.txt`: running text in that language that is not part of the dictionary, used as held-out data.
 * It prints next-character prediction accuracy and the simulated wrong-key rate with and without the adaptation.
 */
class AdaptiveTouchEvaluation {
    private class Layout(val rows: List<String>, val rowOffsets: List<Float>, val keyWidth: Float = 100f, val keyHeight: Float = 130f) {
        val keys = ArrayList<Triple<Int, Float, Float>>() // code point, left, top
        init {
            for ((r, row) in rows.withIndex()) {
                var i = 0
                var column = 0
                while (i < row.length) {
                    val codePoint = row.codePointAt(i)
                    i += Character.charCount(codePoint)
                    keys.add(Triple(codePoint, (rowOffsets[r] + column) * keyWidth, r * keyHeight))
                    column++
                }
            }
        }
        val count get() = keys.size
        val left = IntArray(count) { keys[it].second.roundToInt() }
        val top = IntArray(count) { keys[it].third.roundToInt() }
        val width = IntArray(count) { keyWidth.roundToInt() }
        val height = IntArray(count) { keyHeight.roundToInt() }
        fun indexOf(codePoint: Int) = keys.indexOfFirst { it.first == codePoint }
        /** Plain geometric detection: the key whose hitbox contains the point, as KeyDetector does for a tiled keyboard. */
        fun geometric(x: Float, y: Float): Int {
            for (i in 0 until count) {
                if (x >= left[i] && x < left[i] + width[i] && y >= top[i] && y < top[i] + height[i]) return i
            }
            // outside every key (beyond the row ends): nearest key by edge distance, like edge keys with enlarged hitboxes
            var best = -1
            var bestDistance = Float.MAX_VALUE
            for (i in 0 until count) {
                val dx = maxOf(left[i] - x, 0f, x - (left[i] + width[i]))
                val dy = maxOf(top[i] - y, 0f, y - (top[i] + height[i]))
                val d = dx * dx + dy * dy
                if (d < bestDistance) { bestDistance = d; best = i }
            }
            return best
        }
        /** Keys within 1.2 key widths of the point, like ProximityInfo.getNearestKeys. */
        fun nearest(x: Float, y: Float): List<Int> = (0 until count).filter {
            val cx = left[it] + width[it] / 2f
            val cy = top[it] + height[it] / 2f
            (cx - x) * (cx - x) + (cy - y) * (cy - y) <= (1.2f * keyWidth + keyWidth) * (1.2f * keyWidth + keyWidth)
        }
    }

    private val layouts = mapOf(
        "tr" to Layout(listOf("qwertyuıopğü", "asdfghjklşi", "zxcvbnmöç"), listOf(0f, 0.5f, 1.5f)),
        "en" to Layout(listOf("qwertyuiop", "asdfghjkl", "zxcvbnm"), listOf(0f, 0.5f, 1.5f)),
        "de" to Layout(listOf("qwertzuiopü", "asdfghjklöä", "yxcvbnm"), listOf(0f, 0.5f, 1.5f)),
    )

    @Test fun evaluate() {
        val dir = System.getenv("CHARLM_EVAL_DIR")
        assumeTrue("set CHARLM_EVAL_DIR to run the evaluation", dir != null && File(dir).isDirectory)
        val report = StringBuilder()
        for ((language, layout) in layouts) {
            val wordsFile = File(dir, "words_$language.tsv")
            val textFile = File(dir, "text_$language.txt")
            if (!wordsFile.isFile || !textFile.isFile) continue
            evaluateLanguage(language, layout, wordsFile, textFile, report)
        }
        println(report)
        File(dir, "report.txt").writeText(report.toString())
    }

    private fun evaluateLanguage(language: String, layout: Layout, wordsFile: File, textFile: File, report: StringBuilder) {
        val locale = Locale.forLanguageTag(language)
        val start = System.currentTimeMillis()
        val dictionary = HashMap<String, Int>()
        wordsFile.forEachLine { line ->
            val tab = line.indexOf('\t')
            if (tab > 0) dictionary[line.substring(0, tab)] = line.substring(tab + 1).trim().toIntOrNull() ?: 0
        }
        val history = System.getenv("CHARLM_HISTORY")?.toIntOrNull() ?: CharNgramModel.DEFAULT_MAX_HISTORY
        val weightBase = System.getenv("CHARLM_BASE")?.toDoubleOrNull() ?: CharNgramModel.DEFAULT_WEIGHT_BASE
        val model = CharNgramModel(language, "eval", history)
        for ((word, probability) in dictionary) model.addWord(word, CharNgramModel.dictionaryWeight(probability, weightBase))
        val buildMillis = System.currentTimeMillis() - start
        val runtime = Runtime.getRuntime()
        report.appendLine("== $language: ${dictionary.size} dictionary words, history $history, weight base $weightBase, model $model, built in $buildMillis ms")

        // held-out words: lower case, letters of the layout only
        val layoutChars = layout.keys.map { it.first }.toSet()
        val words = textFile.readText().lowercase(locale).split(Regex("[^\\p{L}']+"))
            .filter { w -> w.length >= 2 && w.all { layoutChars.contains(it.code) } }
        val inDictionary = words.count { dictionary.containsKey(it) }
        report.appendLine("held-out words: ${words.size}, in dictionary: $inDictionary (${percent(inDictionary, words.size)})")

        // next-character prediction accuracy (model only, dictionary completions only, mixed) vs letter frequency
        val letterFrequency = IntArray(layout.count)
        for ((word, probability) in dictionary) for (c in word.lowercase(locale)) {
            val i = layout.indexOf(c.code)
            if (i >= 0) letterFrequency[i] += CharNgramModel.dictionaryWeight(probability, weightBase)
        }
        val sortedDictionary = dictionary.entries.sortedByDescending { it.value }.map { it.key to it.value }
        var positions = 0
        val top1 = IntArray(4)
        val top3 = IntArray(4)
        val cache = HashMap<String, FloatArray?>()
        for (word in words) {
            for (i in word.indices) {
                val prefix = word.substring(0, i)
                val target = layout.indexOf(word[i].code)
                if (target < 0) continue
                positions++
                val modelProbabilities = model.nextSymbolProbabilities(prefix)
                val lm = FloatArray(layout.count) { model.probabilityOfCodePoint(modelProbabilities, layout.keys[it].first) }
                val dict = cache.getOrPut(prefix) { dictionaryCompletionDistribution(sortedDictionary, prefix, layout) }
                val mixed = FloatArray(layout.count) { k -> if (dict == null) lm[k] else 0.7f * dict[k] + 0.3f * lm[k] }
                val unigram = FloatArray(layout.count) { letterFrequency[it].toFloat() }
                for ((m, scores) in listOf(unigram, lm, dict ?: lm, mixed).withIndex()) {
                    val ranked = (0 until layout.count).sortedByDescending { scores[it] }
                    if (ranked[0] == target) top1[m]++
                    if (target in ranked.take(3)) top3[m]++
                }
            }
        }
        report.appendLine("next-char accuracy over $positions positions: letter-frequency top1 ${percent(top1[0], positions)} top3 ${percent(top3[0], positions)}"
                + " | char model top1 ${percent(top1[1], positions)} top3 ${percent(top3[1], positions)}"
                + " | dictionary completions top1 ${percent(top1[2], positions)} top3 ${percent(top3[2], positions)}"
                + " | mixed top1 ${percent(top1[3], positions)} top3 ${percent(top3[3], positions)}")

        // simulated typing: touch = key centre + gaussian noise; decode geometrically vs adaptively
        for (sigma in listOf(0.25f, 0.35f, 0.45f)) {
            val random = Random(42)
            var touches = 0
            var geometricErrors = 0
            val adaptiveErrors = IntArray(4)
            val lambdas = listOf(0.5f, 1f, 1.5f, 2f)
            for (word in words) {
                for (i in word.indices) {
                    val target = layout.indexOf(word[i].code)
                    if (target < 0) continue
                    val prefix = word.substring(0, i)
                    val x = layout.left[target] + layout.width[target] / 2f + (random.nextGaussian() * sigma * layout.keyWidth).toFloat()
                    val y = layout.top[target] + layout.height[target] / 2f + (random.nextGaussian() * sigma * layout.keyHeight).toFloat()
                    touches++
                    val geometric = layout.geometric(x, y)
                    if (geometric != target) geometricErrors++
                    val modelProbabilities = model.nextSymbolProbabilities(prefix)
                    val dict = cache[prefix]
                    val nearest = layout.nearest(x, y)
                    val n = nearest.size
                    val left = IntArray(n) { layout.left[nearest[it]] }
                    val top = IntArray(n) { layout.top[nearest[it]] }
                    val width = IntArray(n) { layout.width[nearest[it]] }
                    val height = IntArray(n) { layout.height[nearest[it]] }
                    val probability = FloatArray(n) {
                        val key = nearest[it]
                        val lm = model.probabilityOfCodePoint(modelProbabilities, layout.keys[key].first)
                        if (dict == null) lm else 0.7f * dict[key] + 0.3f * lm
                    }
                    val geometricIndex = nearest.indexOf(geometric)
                    for ((li, lambda) in lambdas.withIndex()) {
                        val chosen = AdaptiveKeyHitTest.decide(x, y, left, top, width, height, probability, n, geometricIndex, lambda)
                        val key = if (chosen >= 0) nearest[chosen] else geometric
                        if (key != target) adaptiveErrors[li]++
                    }
                }
            }
            report.append("touch noise σ=$sigma key: wrong key geometric ${percent(geometricErrors, touches)}")
            for ((li, lambda) in lambdas.withIndex()) {
                report.append(" | λ=$lambda adaptive ${percent(adaptiveErrors[li], touches)} (${relative(geometricErrors, adaptiveErrors[li])})")
            }
            report.appendLine()
        }

        // cost side: the user deliberately types an unlikely letter (the least likely neighbour), touching
        // 0.3 key widths off centre towards a likely neighbour
        run {
            var cases = 0
            var lost = 0
            val random = Random(7)
            for (word in words.take(3000)) {
                for (i in 1 until word.length) {
                    val prefix = word.substring(0, i)
                    val modelProbabilities = model.nextSymbolProbabilities(prefix)
                    val dict = cache[prefix]
                    val probabilityOf = { key: Int ->
                        val lm = model.probabilityOfCodePoint(modelProbabilities, layout.keys[key].first)
                        if (dict == null) lm else 0.7f * dict[key] + 0.3f * lm
                    }
                    // pick the least likely key in the row of the true key (within 3 keys), tap it off-centre
                    val target = layout.indexOf(word[i].code)
                    if (target < 0) continue
                    val row = layout.top[target]
                    val unlikely = (0 until layout.count).filter { layout.top[it] == row }.minByOrNull { probabilityOf(it) } ?: continue
                    val neighbour = listOf(unlikely - 1, unlikely + 1).filter { it in 0 until layout.count && layout.top[it] == row }
                        .maxByOrNull { probabilityOf(it) } ?: continue
                    val direction = if (neighbour > unlikely) 1f else -1f
                    val x = layout.left[unlikely] + layout.width[unlikely] / 2f + direction * 0.3f * layout.keyWidth
                    val y = layout.top[unlikely] + layout.height[unlikely] / 2f + (random.nextGaussian() * 0.1f * layout.keyHeight).toFloat()
                    val nearest = layout.nearest(x, y)
                    val n = nearest.size
                    val chosen = AdaptiveKeyHitTest.decide(x, y,
                        IntArray(n) { layout.left[nearest[it]] }, IntArray(n) { layout.top[nearest[it]] },
                        IntArray(n) { layout.width[nearest[it]] }, IntArray(n) { layout.height[nearest[it]] },
                        FloatArray(n) { probabilityOf(nearest[it]) }, n, nearest.indexOf(unlikely), 1f)
                    cases++
                    if (chosen < 0 || nearest[chosen] != unlikely) lost++
                }
            }
            report.appendLine("deliberate unlikely letter tapped 0.3 key off-centre towards a likely neighbour (λ=1): lost ${percent(lost, cases)} of $cases")
        }
        // ---- word level: can the character model help with words the dictionary does not know? ----
        // 1) autocorrect guard: an unknown valid word vs a typo of a known word; both have a dictionary word nearby that
        //    autocorrect would propose. Score = log P(typed) - log P(proposed) per character.
        // 2) completions: for unknown words, is the true word among the model's top completions of its prefix?
        run {
            val unknownWords = words.filter { it.length >= 4 && !dictionary.containsKey(it) }.distinct().take(600)
            val knownWords = words.filter { it.length >= 4 && dictionary.containsKey(it) }.distinct().take(600)
            val byPrefix2 = HashMap<String, ArrayList<String>>()
            for ((word, _) in dictionary) if (word.length >= 3) byPrefix2.getOrPut(word.substring(0, 2)) { ArrayList() }.add(word)
            fun nearestDictionaryWord(typed: String): String? {
                if (typed.length < 3) return null
                val candidates = byPrefix2[typed.substring(0, 2)] ?: return null
                var best: String? = null
                var bestDistance = 3
                for (c in candidates) {
                    if (c == typed || abs(c.length - typed.length) > 2) continue
                    val d = editDistance(typed, c, bestDistance)
                    if (d < bestDistance || (d == bestDistance && best != null && (dictionary[c] ?: 0) > (dictionary[best] ?: 0))) {
                        bestDistance = d; best = c
                    }
                }
                return best
            }
            val random = Random(11)
            fun typo(word: String): String {
                val i = 1 + random.nextInt(word.length - 1)
                return when (random.nextInt(3)) {
                    0 -> word.substring(0, i) + word.substring(i + 1) // drop
                    1 -> if (i + 1 < word.length) word.substring(0, i) + word[i + 1] + word[i] + word.substring(i + 2) else word.substring(0, i - 1) + word[i] + word[i - 1] // swap
                    else -> { // neighbouring key
                        val k = layout.indexOf(word[i].code)
                        val row = if (k >= 0) layout.top[k] else -1
                        val neighbour = listOf(k - 1, k + 1).filter { it in 0 until layout.count && layout.top[it] == row }.let { if (it.isEmpty()) null else it[random.nextInt(it.size)] }
                        if (neighbour == null) word else word.substring(0, i) + String(Character.toChars(layout.keys[neighbour].first)) + word.substring(i + 1)
                    }
                }
            }
            val validScores = ArrayList<Double>()
            val typoScores = ArrayList<Double>()
            for (w in unknownWords) {
                val proposed = nearestDictionaryWord(w) ?: continue
                validScores.add((model.logProbabilityOfWord(w) - model.logProbabilityOfWord(proposed)) / w.length)
            }
            for (w in knownWords) {
                val t = typo(w)
                if (t == w || dictionary.containsKey(t)) continue
                val proposed = nearestDictionaryWord(t) ?: continue
                if (proposed != w) continue // the engine would propose something else; ignore
                typoScores.add((model.logProbabilityOfWord(t) - model.logProbabilityOfWord(proposed)) / t.length)
            }
            report.appendLine("autocorrect guard: ${validScores.size} unknown valid words with a dictionary word within 2 edits, ${typoScores.size} typos of known words")
            for (threshold in listOf(-0.6, -0.4, -0.3, -0.2, -0.1, 0.0)) {
                val keepValid = validScores.count { it >= threshold }
                val keepTypo = typoScores.count { it >= threshold }
                report.appendLine("  guard at $threshold: keeps ${percent(keepValid, validScores.size)} of valid unknown words from autocorrect, but also ${percent(keepTypo, typoScores.size)} of typos")
            }
            var cases = 0
            val hit = IntArray(3)
            for (w in unknownWords) {
                for (k in listOf(3, 4, 5)) {
                    if (w.length <= k + 1) continue
                    cases++
                    val completions = model.complete(w.substring(0, k), 3).map { it.first }
                    val i = completions.indexOf(w)
                    if (i >= 0) for (j in i until 3) hit[j]++
                }
            }
            report.appendLine("completions of unknown words from 3/4/5 letters: top1 ${percent(hit[0], cases)} top3 ${percent(hit[2], cases)} of $cases")
            cases = 0; hit.fill(0)
            for (w in knownWords) {
                for (k in listOf(3, 4, 5)) {
                    if (w.length <= k + 1) continue
                    cases++
                    val completions = model.complete(w.substring(0, k), 3).map { it.first }
                    val i = completions.indexOf(w)
                    if (i >= 0) for (j in i until 3) hit[j]++
                }
            }
            report.appendLine("completions of known words from 3/4/5 letters: top1 ${percent(hit[0], cases)} top3 ${percent(hit[2], cases)} of $cases")
        }
        report.appendLine("memory used after evaluation: ${(runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)} MB")
        report.appendLine()
    }

    /** Distribution of next letters over the layout from the 18 most probable dictionary words with the prefix, rank-decayed like Suggest does. */
    private fun dictionaryCompletionDistribution(sortedDictionary: List<Pair<String, Int>>, prefix: String, layout: Layout): FloatArray? {
        val weights = FloatArray(layout.count)
        var weight = 1f
        var found = 0
        for ((word, _) in sortedDictionary) {
            if (word.length <= prefix.length || !word.startsWith(prefix)) continue
            val key = layout.indexOf(word.codePointAt(prefix.length))
            if (key < 0) continue
            weights[key] += weight
            weight *= NextCharPredictor.RANK_DECAY
            if (++found >= 18) break
        }
        if (found == 0) return null
        val sum = weights.sum()
        for (i in weights.indices) weights[i] /= sum
        return weights
    }

    /** Levenshtein distance, giving up (returning [limit]) once it cannot be below [limit]. */
    private fun editDistance(a: String, b: String, limit: Int): Int {
        if (abs(a.length - b.length) >= limit) return limit
        var previous = IntArray(b.length + 1) { it }
        var current = IntArray(b.length + 1)
        for (i in 1..a.length) {
            current[0] = i
            var rowMin = i
            for (j in 1..b.length) {
                val cost = if (a[i - 1] == b[j - 1]) 0 else 1
                current[j] = min(min(previous[j] + 1, current[j - 1] + 1), previous[j - 1] + cost)
                rowMin = min(rowMin, current[j])
            }
            if (rowMin >= limit) return limit
            val t = previous; previous = current; current = t
        }
        return min(previous[b.length], limit)
    }

    private fun percent(part: Int, total: Int) = if (total == 0) "n/a" else "%.2f%%".format(Locale.US, 100.0 * part / total)
    private fun relative(before: Int, after: Int) = if (before == 0) "n/a" else "%+.0f%%".format(Locale.US, 100.0 * (after - before) / before)
}
