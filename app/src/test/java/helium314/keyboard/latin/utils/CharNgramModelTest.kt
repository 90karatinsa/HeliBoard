// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class CharNgramModelTest {
    private val turkishWords = listOf(
        "geliyorum" to 90, "geliyorsun" to 70, "geliyor" to 120, "gidiyorum" to 85, "gidiyoruz" to 80,
        "yapıyorum" to 88, "biliyorum" to 100, "istiyorum" to 95, "evlerimiz" to 60, "evler" to 110,
        "evde" to 120, "okulda" to 110, "okula" to 100, "kitaplar" to 90, "kitapları" to 80,
        "arkadaşlar" to 95, "arkadaşım" to 90, "İstanbul" to 130, "ılık" to 60, "ışık" to 90,
    )

    private fun turkishModel(): CharNgramModel {
        val model = CharNgramModel("tr", "test")
        for ((word, probability) in turkishWords) model.addWord(word, CharNgramModel.dictionaryWeight(probability))
        return model
    }

    @Test fun `distribution sums to one for seen and unseen prefixes`() {
        val model = turkishModel()
        for (prefix in listOf("", "g", "geliyo", "xyz", "kitapl", "zzzzzz")) {
            val probabilities = model.nextSymbolProbabilities(prefix)
            assertEquals(1f, probabilities.sum(), 1e-4f, "prefix '$prefix'")
            assertTrue(probabilities.all { it > 0f }, "every symbol keeps some probability for '$prefix'")
        }
    }

    @Test fun `predicts the continuation of a common suffix`() {
        val model = turkishModel()
        val afterGeliyo = model.nextSymbolProbabilities("geliyo")
        assertEquals('r'.code, model.codePointOf(afterGeliyo.indices.maxBy { afterGeliyo[it] }))
        // "-yor" is followed by "u" (yorum / yoruz) or "s" (yorsun) in the training data, never by "e"
        val afterYor = model.nextSymbolProbabilities("geliyor")
        assertTrue(afterYor[model.symbolOf('u'.code)] > afterYor[model.symbolOf('e'.code)])
        // unseen stem, but a seen suffix pattern: the model generalizes the suffix
        val unseen = model.nextSymbolProbabilities("koşuyo")
        assertEquals('r'.code, model.codePointOf(unseen.indices.maxBy { unseen[it] }))
    }

    @Test fun `turkish case folding keeps dotted and dotless i apart`() {
        val model = turkishModel()
        assertEquals(model.symbolOf('i'.code), model.symbolOf('İ'.code))
        assertEquals(model.symbolOf('ı'.code), model.symbolOf('I'.code))
        assertTrue(model.symbolOf('i'.code) != model.symbolOf('ı'.code))
        // "İstanbul" was added capitalized and must be found via the lower case prefix
        val afterIst = model.nextSymbolProbabilities("ist")
        assertEquals('a'.code, model.codePointOf(afterIst.indices.maxBy { afterIst[it] }))
    }

    @Test fun `english model lower cases with english rules`() {
        val model = CharNgramModel("en", "test")
        model.addWord("Istanbul", 100)
        model.addWord("island", 100)
        assertEquals(model.symbolOf('i'.code), model.symbolOf('I'.code))
    }

    @Test fun `words with digits or symbols are ignored but apostrophes and hyphens are allowed`() {
        val model = CharNgramModel("en", "test")
        model.addWord("abc123", 10)
        model.addWord("hello!", 10)
        model.addWord("😀", 10)
        assertEquals(0, model.wordCount)
        model.addWord("don't", 10)
        model.addWord("well-known", 10)
        assertEquals(2, model.wordCount)
        assertTrue(model.symbolOf('\''.code) >= 2)
    }

    @Test fun `unknown characters get a share of the other symbol`() {
        val model = turkishModel()
        val probabilities = model.nextSymbolProbabilities("ge")
        val unknown = model.probabilityOfCodePoint(probabilities, 'w'.code)
        assertTrue(unknown >= 0f && unknown < probabilities[CharNgramModel.OTHER])
        assertEquals(-1, model.symbolOf('1'.code))
    }

    @Test fun `learning a word changes the prediction`() {
        val model = turkishModel()
        val before = model.nextCodePointProbability("kitaplarım", 'ı'.code)
        repeat(3) { model.addWord("kitaplarımı", CharNgramModel.USER_WORD_WEIGHT) }
        val after = model.nextCodePointProbability("kitaplarım", 'ı'.code)
        assertTrue(after > before, "learned continuation should become more likely ($before -> $after)")
    }

    @Test fun `serialization round trip preserves predictions`() {
        val model = turkishModel()
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { model.writeTo(it) }
        val copy = DataInputStream(ByteArrayInputStream(bytes.toByteArray())).use { CharNgramModel.read(it) }
        assertEquals(model.dictKey, copy.dictKey)
        assertEquals(model.localeTag, copy.localeTag)
        assertEquals(model.symbolCount, copy.symbolCount)
        assertEquals(model.size, copy.size)
        assertEquals(model.wordCount, copy.wordCount)
        for (prefix in listOf("", "gel", "geliyo", "kitapl", "arkada", "q")) {
            val a = model.nextSymbolProbabilities(prefix)
            val b = copy.nextSymbolProbabilities(prefix)
            for (i in a.indices) assertEquals(a[i], b[i], 1e-6f, "prefix '$prefix' symbol $i")
        }
        assertEquals(model.symbolOf('ş'.code), copy.symbolOf('ş'.code))
    }

    @Test fun `rejects foreign files`() {
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { it.writeInt(12345); it.writeInt(1) }
        val result = runCatching { DataInputStream(ByteArrayInputStream(bytes.toByteArray())).use { CharNgramModel.read(it) } }
        assertTrue(result.isFailure)
    }

    @Test fun `usefulness requires enough words`() {
        val model = CharNgramModel("en", "test")
        assertFalse(model.isUseful)
        for (i in 0 until 300) model.addWord("word$i".filter { it.isLetter() } + "abcdefghij"[i % 10], 1)
        assertTrue(model.isUseful)
    }

    @Test fun `completion finds likely word endings`() {
        val model = turkishModel()
        val completions = model.complete("geliyo", 3)
        assertTrue(completions.isNotEmpty())
        assertTrue(completions.all { it.first.startsWith("geliyo") })
        assertTrue(completions.any { it.first == "geliyor" || it.first == "geliyorum" }, "got $completions")
        assertTrue(model.complete("", 3).isEmpty())
    }

    @Test fun `word plausibility is higher for well formed words`() {
        val model = turkishModel()
        assertTrue(model.logProbabilityOfWord("geliyorlar") > model.logProbabilityOfWord("gkzpqrtw"))
        assertNull(model.logProbabilityOfWord("abc1").takeIf { it.isFinite() })
    }

    @Test fun `dictionary weight grows with probability and is at least one`() {
        assertEquals(1, CharNgramModel.dictionaryWeight(0))
        assertEquals(1, CharNgramModel.dictionaryWeight(-5))
        assertTrue(CharNgramModel.dictionaryWeight(120) > CharNgramModel.dictionaryWeight(60))
        assertTrue(CharNgramModel.dictionaryWeight(255) < 1_000_000)
    }

    @Test fun `counts beyond the range of an int stay consistent`() {
        val model = CharNgramModel("tr", "test")
        val weight = CharNgramModel.dictionaryWeight(255)
        repeat(Int.MAX_VALUE / weight + 2) { model.addWord("aaa", weight) }
        val probabilities = model.nextSymbolProbabilities("a")
        assertEquals(1f, probabilities.sum(), 1e-4f)
        assertTrue(probabilities[model.symbolOf('a'.code)] > 0.9f, "the only continuation seen dominates")
        assertTrue(model.totalWeight > Int.MAX_VALUE)
    }
}
