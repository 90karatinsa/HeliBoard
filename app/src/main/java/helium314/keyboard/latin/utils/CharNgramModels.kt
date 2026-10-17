// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import android.content.Context
import android.os.Process
import helium314.keyboard.latin.dictionary.Dictionary
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Keeps one [CharNgramModel] per locale, builds them in the background from the loaded dictionaries, caches them on
 * disk so later starts are fast, and feeds words the user types into them.
 */
object CharNgramModels {
    private const val TAG = "CharNgramModels"
    private const val MAX_MODELS = 3
    private const val CACHE_DIR = "charlm"

    private val historySeeded = HashSet<String>()
    /** Tags of models that hold words of the user (learned or seeded from the user history). */
    private val withUserWords = HashSet<String>()
    private val inProgress = HashSet<String>()
    /** Requests that arrived while a build for the same locale was running; they are re-evaluated when it ends. */
    private val pending = HashMap<String, Request>()
    private val models = object : LinkedHashMap<String, CharNgramModel>(MAX_MODELS + 1, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, CharNgramModel>?): Boolean {
            val evict = size > MAX_MODELS
            if (evict && eldest != null) {
                historySeeded.remove(eldest.key) // a reloaded model needs the user's words again
                withUserWords.remove(eldest.key)
            }
            return evict
        }
    }

    // One thread at background priority: building then competes neither with the suggestion worker the UI thread
    // waits for nor with other builds, and user history seeding (which blocks on the dictionary's executor) stays
    // off the shared pools.
    private val dispatcher = Executors.newSingleThreadExecutor { runnable ->
        Thread({
            try {
                Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
            } catch (e: Throwable) {
                // not running on Android
            }
            runnable.run()
        }, "CharNgramModels")
    }.asCoroutineDispatcher()
    private val scope = CoroutineScope(dispatcher + SupervisorJob())

    private class Request(val context: Context, val locale: Locale, val dictKey: String, val mainDict: Dictionary?, val userHistoryDict: Dictionary?)

    /** Incremented whenever a model is added, replaced or removed; lets callers invalidate derived caches cheaply. */
    @Volatile
    var version = 0
        private set

    /** The model for [locale], or null if none has been built (yet). Cheap, safe to call on the UI thread. */
    @JvmStatic
    @Synchronized
    fun get(locale: Locale): CharNgramModel? = models[locale.toLanguageTag()]

    /**
     * Makes sure a model built from the current dictionaries of [locale] exists, building or loading it in the
     * background if necessary. [dictKey] identifies the dictionary files (see [dictKeyFor]); a model built from
     * other files is replaced. Words of [userHistoryDict] are added once per model so the model knows the user's
     * vocabulary from the start. Nothing is built for scripts the prediction does not support.
     */
    fun ensureBuilt(context: Context, locale: Locale, dictKey: String, mainDict: Dictionary?, userHistoryDict: Dictionary?) {
        if (!NextCharPredictor.isSupportedLocale(locale)) return
        val tag = locale.toLanguageTag()
        val request = Request(context.applicationContext, locale, dictKey, mainDict, userHistoryDict)
        synchronized(this) {
            val existing = models[tag]
            val upToDate = existing != null && existing.dictKey == dictKey
            val needsHistory = userHistoryDict != null && !historySeeded.contains(tag)
            if (upToDate && !needsHistory) return
            if (inProgress.contains(tag)) {
                pending[tag] = request
                return
            }
            inProgress.add(tag)
        }
        scope.launch { process(request) }
    }

    private fun process(request: Request) {
        val tag = request.locale.toLanguageTag()
        try {
            var model = synchronized(this) { models[tag] }
            if (model == null || model.dictKey != request.dictKey) {
                model = loadFromCache(request.context, tag, request.dictKey)
                    ?: buildFromDictionary(request.context, tag, request.dictKey, request.mainDict)
                val seeded = model != null && request.userHistoryDict != null && seedUserHistory(model, request.userHistoryDict)
                synchronized(this) {
                    if (model != null) models[tag] = model else models.remove(tag)
                    if (seeded) historySeeded.add(tag) else historySeeded.remove(tag)
                    version++
                }
            } else if (request.userHistoryDict != null && synchronized(this) { !historySeeded.contains(tag) }) {
                if (seedUserHistory(model, request.userHistoryDict)) synchronized(this) { historySeeded.add(tag) }
            }
        } catch (e: Throwable) {
            Log.e(TAG, "could not build character model for $tag", e)
        } finally {
            val next = synchronized(this) {
                inProgress.remove(tag)
                pending.remove(tag)
            }
            if (next != null) ensureBuilt(next.context, next.locale, next.dictKey, next.mainDict, next.userHistoryDict)
        }
    }

    /** Adds a word the user typed to the model of [locale], if there is one. Cheap, safe on the UI thread. */
    @JvmStatic
    fun learn(locale: Locale, word: String) {
        val model = get(locale) ?: return
        model.addWord(word, CharNgramModel.USER_WORD_WEIGHT)
        synchronized(this) { withUserWords.add(model.localeTag) }
    }

    /**
     * Forgets the user's words: drops the in-memory model of [locale] if it holds any, so that the next [ensureBuilt]
     * reloads it from the cache. Called whenever the user history is cleared, which happens often while personalized
     * suggestions are off, so a model without user words is kept.
     */
    @JvmStatic
    @Synchronized
    fun forget(locale: Locale) {
        val tag = locale.toLanguageTag()
        if (withUserWords.remove(tag) && models.remove(tag) != null) version++
        historySeeded.remove(tag)
    }

    /** A key describing the dictionary files currently used for [locale] and the model parameters, for cache validation. */
    fun dictKeyFor(context: Context, locale: Locale): String {
        val files = DictionaryInfoUtils.getCachedDictsForLocale(locale, context)
        val description = files.map { "${it.name}:${it.length()}:${it.lastModified()}" }.sorted().joinToString(";")
        return "${files.size}-${Integer.toHexString(description.hashCode())}-h${CharNgramModel.DEFAULT_MAX_HISTORY}-b${CharNgramModel.DEFAULT_WEIGHT_BASE}"
    }

    @Synchronized
    internal fun setModelForTests(locale: Locale, model: CharNgramModel?) {
        val tag = locale.toLanguageTag()
        if (model != null) models[tag] = model else models.remove(tag)
        historySeeded.remove(tag)
        withUserWords.remove(tag)
        version++
    }

    private fun cacheFile(context: Context, tag: String): File {
        val dir = File(context.filesDir, CACHE_DIR)
        if (!dir.isDirectory) dir.mkdirs()
        return File(dir, "$tag.bin")
    }

    private fun loadFromCache(context: Context, tag: String, dictKey: String): CharNgramModel? {
        val file = cacheFile(context, tag)
        if (!file.isFile) return null
        return try {
            val start = System.currentTimeMillis()
            val model = DataInputStream(BufferedInputStream(FileInputStream(file), 1 shl 16)).use { CharNgramModel.read(it) }
            if (model.dictKey != dictKey || model.localeTag != tag) {
                Log.i(TAG, "cached character model for $tag is outdated")
                file.delete()
                return null
            }
            Log.i(TAG, "loaded $model from cache in ${System.currentTimeMillis() - start} ms")
            model
        } catch (e: IOException) {
            Log.w(TAG, "could not read cached character model for $tag, deleting it", e)
            file.delete()
            null
        }
    }

    private fun buildFromDictionary(context: Context, tag: String, dictKey: String, mainDict: Dictionary?): CharNgramModel? {
        if (mainDict == null) return null
        val start = System.currentTimeMillis()
        val model = CharNgramModel(tag, dictKey)
        val complete = mainDict.forEachWord { word, probability ->
            // negative: not a word (e.g. a shortcut source), only in the dictionary to be corrected
            if (probability >= 0) model.addWord(word, CharNgramModel.dictionaryWeight(probability))
        }
        if (!complete) {
            Log.i(TAG, "dictionary of $tag was closed or busy while building the character model, not using the partial model")
            return null
        }
        Log.i(TAG, "built $model in ${System.currentTimeMillis() - start} ms")
        if (!model.isUseful) {
            Log.i(TAG, "character model for $tag is not useful, not using it")
            return null
        }
        try {
            val file = cacheFile(context, tag)
            val temp = File(file.path + ".tmp")
            DataOutputStream(BufferedOutputStream(FileOutputStream(temp), 1 shl 16)).use { model.writeTo(it) }
            if (!temp.renameTo(file)) temp.delete()
        } catch (e: IOException) {
            Log.w(TAG, "could not write character model cache for $tag", e)
        }
        return model
    }

    /** Adds the user's words to [model]; false if the dictionary could not be read completely. */
    private fun seedUserHistory(model: CharNgramModel, userHistoryDict: Dictionary): Boolean {
        val start = System.currentTimeMillis()
        var count = 0
        val complete = userHistoryDict.forEachWord { word, _ ->
            model.addWord(word, CharNgramModel.USER_WORD_WEIGHT)
            count++
        }
        if (count > 0) synchronized(this) { withUserWords.add(model.localeTag) }
        Log.i(TAG, "added $count user history words to ${model.localeTag} model in ${System.currentTimeMillis() - start} ms" +
                if (complete) "" else " (incomplete)")
        return complete
    }
}
