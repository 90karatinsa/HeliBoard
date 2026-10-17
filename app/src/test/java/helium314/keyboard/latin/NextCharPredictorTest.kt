// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin

import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.CoordinateUtils
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.utils.CharNgramModel
import helium314.keyboard.latin.utils.CharNgramModels
import helium314.keyboard.latin.utils.NextCharPredictor
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.Locale
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// Robolectric because WordComposer uses android.text.SpannableStringBuilder
@RunWith(RobolectricTestRunner::class)
class NextCharPredictorTest {
    private val turkish = Locale.forLanguageTag("tr")
    private val composer = WordComposer()
    private val state = FakeInputState(composer)

    private class FakeInputState(private val composer: WordComposer) : NextCharPredictor.InputState {
        override val wordComposer get() = composer
        override var locale: Locale? = null
        override var isCursorAtWordStart = true
        override fun isWordSeparator(code: Int) = code == ' '.code
    }

    @BeforeTest fun setUp() {
        composer.reset()
        state.locale = null
        state.isCursorAtWordStart = true
        NextCharPredictor.setInputState(state)
        NextCharPredictor.onComposingReset()
        CharNgramModels.setModelForTests(turkish, null)
        CharNgramModels.setModelForTests(Locale.ENGLISH, null)
    }

    @AfterTest fun tearDown() {
        NextCharPredictor.setInputState(null)
        CharNgramModels.setModelForTests(turkish, null)
        CharNgramModels.setModelForTests(Locale.ENGLISH, null)
    }

    /** Puts [word] into the composer the way resuming a word does, cursor at its end. */
    private fun type(word: String) {
        if (word.isEmpty()) {
            composer.reset()
            return
        }
        val codePoints = StringUtils.toCodePointArray(word)
        composer.setComposingWord(codePoints, CoordinateUtils.newCoordinateArray(codePoints.size, Constants.NOT_A_COORDINATE, Constants.NOT_A_COORDINATE))
    }

    @Test fun `nothing is known without input state or sources`() {
        NextCharPredictor.setInputState(null)
        assertNull(NextCharPredictor.currentDistribution(turkish))
        NextCharPredictor.setInputState(state)
        assertNull(NextCharPredictor.currentDistribution(turkish), "no model, nothing published")
    }

    @Test fun `published weights are used only for the matching prefix and generation`() {
        type("ge")
        val generation = NextCharPredictor.getGeneration()
        NextCharPredictor.publish("ge", generation, mapOf('l'.code to 3f, 'n'.code to 1f), 2, turkish)
        val distribution = assertNotNull(NextCharPredictor.currentDistribution(turkish))
        assertTrue(distribution.hasDictionaryEvidence)
        assertEquals(0.75f, distribution.probabilityOf('l'.code), 1e-5f)
        assertEquals(0.25f, distribution.probabilityOf('n'.code), 1e-5f)
        assertEquals(0f, distribution.probabilityOf('x'.code), 1e-5f)

        type("gel") // the user typed on; the publication is stale now
        assertNull(NextCharPredictor.currentDistribution(turkish))

        type("ge") // back to the prefix the publication was made for: valid again
        assertNotNull(NextCharPredictor.currentDistribution(turkish))

        NextCharPredictor.onComposingReset() // a commit happened: the publication must not be used any more
        type("ge")
        assertNull(NextCharPredictor.currentDistribution(turkish))
    }

    @Test fun `a publication tagged with an older generation is rejected`() {
        type("")
        val before = NextCharPredictor.getGeneration()
        NextCharPredictor.onComposingReset()
        // suggestions requested before the commit, delivered after it
        NextCharPredictor.publish("", before, mapOf('b'.code to 1f), 1, turkish)
        assertNull(NextCharPredictor.currentDistribution(turkish))
        NextCharPredictor.publish("", NextCharPredictor.getGeneration(), mapOf('b'.code to 1f), 1, turkish)
        assertNotNull(NextCharPredictor.currentDistribution(turkish))
    }

    @Test fun `empty prefix predictions are invalidated by a commit`() {
        type("")
        NextCharPredictor.publish("", NextCharPredictor.getGeneration(), mapOf('b'.code to 1f), 1, turkish)
        assertNotNull(NextCharPredictor.currentDistribution(turkish))
        NextCharPredictor.onComposingReset()
        assertNull(NextCharPredictor.currentDistribution(turkish))
    }

    @Test fun `cursor inside a word that is not composed gives nothing`() {
        type("")
        NextCharPredictor.publish("", NextCharPredictor.getGeneration(), mapOf('b'.code to 1f), 1, turkish)
        state.isCursorAtWordStart = false
        assertNull(NextCharPredictor.currentDistribution(turkish), "a letter typed into a word is not a word start")
        state.isCursorAtWordStart = true
        assertNotNull(NextCharPredictor.currentDistribution(turkish))
        type("ab") // composing: the cursor check does not apply
        state.isCursorAtWordStart = false
        NextCharPredictor.publish("ab", NextCharPredictor.getGeneration(), mapOf('c'.code to 1f), 1, turkish)
        assertNotNull(NextCharPredictor.currentDistribution(turkish))
    }

    @Test fun `keys held down by older touches extend the prefix`() {
        type("ge")
        NextCharPredictor.publish("gel", NextCharPredictor.getGeneration(), mapOf('i'.code to 1f), 1, turkish)
        assertNull(NextCharPredictor.currentDistribution(turkish), "published for gel, but ge is typed")
        val withPendingL = assertNotNull(NextCharPredictor.currentDistribution(turkish, intArrayOf('l'.code)))
        assertEquals("gel", withPendingL.prefix)
        assertEquals("gel", assertNotNull(NextCharPredictor.currentDistribution(turkish, intArrayOf('L'.code))).prefix, "folded with the locale")
        assertNull(NextCharPredictor.currentDistribution(turkish, intArrayOf(' '.code)), "a separator starts a new word")
        assertNull(NextCharPredictor.currentDistribution(turkish, intArrayOf(KeyCode.DELETE)), "unknown effect on the word")
        NextCharPredictor.publish("", NextCharPredictor.getGeneration(), mapOf('b'.code to 1f), 1, turkish)
        assertEquals("", assertNotNull(NextCharPredictor.currentDistribution(turkish, intArrayOf(' '.code))).prefix)
    }

    @Test fun `shifted keys are folded with the locale`() {
        type("")
        NextCharPredictor.publish("", NextCharPredictor.getGeneration(), mapOf('ı'.code to 3f, 'i'.code to 1f), 2, turkish)
        val distribution = assertNotNull(NextCharPredictor.currentDistribution(turkish))
        assertEquals(0.75f, distribution.probabilityOf('I'.code), 1e-5f, "Turkish capital I is dotless ı")
        assertEquals(0.25f, distribution.probabilityOf('İ'.code), 1e-5f, "Turkish capital İ is dotted i")
        assertEquals(0.25f, distribution.probabilityOf('i'.code), 1e-5f)
    }

    @Test fun `publishing folds and normalizes weights and drops non letters`() {
        type("a")
        NextCharPredictor.publish("a", NextCharPredictor.getGeneration(),
            mapOf('B'.code to 1f, 'b'.code to 1f, '1'.code to 5f, ' '.code to 5f), 2, Locale.ENGLISH)
        val distribution = assertNotNull(NextCharPredictor.currentDistribution(Locale.ENGLISH))
        assertEquals(1f, distribution.probabilityOf('b'.code), 1e-5f)
        assertEquals(1f, distribution.probabilityOf('B'.code), 1e-5f)
    }

    @Test fun `batch mode and no evidence give nothing`() {
        type("ab")
        NextCharPredictor.publish("ab", NextCharPredictor.getGeneration(), mapOf('c'.code to 1f), 1, turkish)
        assertNotNull(NextCharPredictor.currentDistribution(turkish))
        NextCharPredictor.publish("ab", NextCharPredictor.getGeneration(), emptyMap(), 0, turkish)
        assertNull(NextCharPredictor.currentDistribution(turkish))
        NextCharPredictor.publish("ab", NextCharPredictor.getGeneration(), mapOf('c'.code to 1f), 1, turkish)
        NextCharPredictor.clearPublished()
        assertNull(NextCharPredictor.currentDistribution(turkish))
    }

    @Test fun `distribution is cached while nothing changes`() {
        type("ab")
        NextCharPredictor.publish("ab", NextCharPredictor.getGeneration(), mapOf('c'.code to 1f), 1, turkish)
        val first = NextCharPredictor.currentDistribution(turkish)
        assertTrue(first === NextCharPredictor.currentDistribution(turkish))
        NextCharPredictor.publish("ab", NextCharPredictor.getGeneration(), mapOf('d'.code to 1f), 1, turkish)
        assertFalse(first === NextCharPredictor.currentDistribution(turkish))
    }

    @Test fun `the language being typed selects the model`() {
        val turkishModel = CharNgramModel("tr", "test")
        val englishModel = CharNgramModel("en", "test")
        repeat(300) { turkishModel.addWord("ağaç", 1); englishModel.addWord("jazz", 1) }
        CharNgramModels.setModelForTests(turkish, turkishModel)
        CharNgramModels.setModelForTests(Locale.ENGLISH, englishModel)
        type("")
        // keyboard is English, the dictionaries say the user writes Turkish
        state.locale = turkish
        val asTurkish = assertNotNull(NextCharPredictor.currentDistribution(Locale.ENGLISH))
        assertTrue(asTurkish.probabilityOf('a'.code) > asTurkish.probabilityOf('j'.code))
        state.locale = null // unknown: the keyboard's language
        val asEnglish = assertNotNull(NextCharPredictor.currentDistribution(Locale.ENGLISH))
        assertTrue(asEnglish.probabilityOf('j'.code) > asEnglish.probabilityOf('a'.code))
        assertFalse(asTurkish === asEnglish)
    }

    @Test fun `forgetting drops only a model that holds the user's words`() {
        val model = CharNgramModel("tr", "test")
        repeat(300) { model.addWord("ağaç", 1) }
        CharNgramModels.setModelForTests(turkish, model)
        CharNgramModels.forget(turkish)
        assertTrue(model === CharNgramModels.get(turkish), "no user words: the model built from the dictionary stays")
        CharNgramModels.learn(turkish, "merhaba")
        CharNgramModels.forget(turkish)
        assertNull(CharNgramModels.get(turkish), "the learned word must not survive clearing the user history")
    }

    @Test fun `unsupported scripts are rejected`() {
        assertTrue(NextCharPredictor.isSupportedLocale(turkish))
        assertTrue(NextCharPredictor.isSupportedLocale(Locale.forLanguageTag("ru")))
        assertFalse(NextCharPredictor.isSupportedLocale(Locale.KOREAN))
        assertFalse(NextCharPredictor.isSupportedLocale(Locale.forLanguageTag("hi")))
        assertFalse(NextCharPredictor.isSupportedLocale(Locale.forLanguageTag("th")))
    }
}
