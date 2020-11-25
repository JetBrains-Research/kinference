package io.kinference.algorithms.completion.tokenizer

import com.github.benmanes.caffeine.cache.Cache
import io.kinference.algorithms.completion.loader.CompletionModelLoader
import io.kinference.algorithms.completion.utils.Caching

internal class BPETokenizer(loader: CompletionModelLoader) {
    private val encodeCache: Cache<String, IntArray> = Caching.default()
    private val decodeCache: Cache<IntArray, String> = Caching.default()

    private data class MergeCandidate(val word1: String, val word2: String) {
        val merged: String = word1 + word2
    }

    private data class MergeCandidateInfo(val vocabIndex: Int?, val position: Int, val mergePair: MergeCandidate)

    private val vocab: MutableMap<String, Int> = HashMap()
    private val reversedVocab: MutableMap<Int, String> = HashMap()
    private val codes: MutableMap<MergeCandidate, Int> = HashMap()
    private val reversedCodes: MutableMap<String, MergeCandidate> = HashMap()

    val eosTokenId = 50256
    val vocabSize: Int
        get() = vocab.size

    init {
        for (pair in loader.getVocabulary().entries) {
            val value = pair.value
            vocab[pair.key] = value
            reversedVocab[value] = pair.key
        }

        for ((left, right) in loader.getMerges()) {
            val pair = MergeCandidate(left, right)
            codes[pair] = codes.size
            reversedCodes[left + right] = pair
        }
    }

    private fun preprocess(s: String): String {
        return s.replace(' ', 'Ġ')
    }

    private fun postprocess(s: String): String {
        return s.replace('Ġ', ' ')
    }

    private fun internalTokenize(word: String): List<String> {
        if (word.isEmpty()) return emptyList()
        if (word.length == 1) return listOf(word)

        var pieces = MutableList(word.length) { "" + word[it] }

        while (pieces.size > 1) {
            val merges = ArrayList<MergeCandidateInfo>(pieces.size)
            for (i in 0 until pieces.size - 1) {
                val current = pieces[i]
                val next = pieces[i + 1]
                val merge = MergeCandidate(current, next)
                if (merge in codes) {
                    merges.add(MergeCandidateInfo(codes[merge], i, merge))
                }
            }
            if (merges.isEmpty()) {
                break
            }

            merges.sortWith(compareBy({ it.vocabIndex }, { it.position }))
            val bigram = merges[0].mergePair
            val positions = merges.filter { it.mergePair == bigram }.map { it.position }

            var i = 0
            val newPieces: MutableList<String> = ArrayList()
            val strBigram = bigram.merged
            for (j in positions) {
                if (j < i) continue

                for (k in i until j) newPieces.add(pieces[k])
                newPieces.add(strBigram)
                i = j + 2
            }
            for (k in i until pieces.size) newPieces.add(pieces[k])
            pieces = newPieces
        }

        if (vocab.isNotEmpty()) {
            pieces = checkVocabAndSplit(pieces)
        }

//        cache[orig] = word
        return pieces
    }

    fun encode(s: String): IntArray {
        return encodeCache.get(s) {
            val tokens = internalTokenize(preprocess(s))
            IntArray(tokens.size) { vocab.getOrDefault(tokens[it], 0) }
        }!!
    }

    fun decode(ids: IntArray): String {
        return decodeCache.get(ids) {
            postprocess(ids.joinToString("") { reversedVocab.getOrDefault(it, "") })
        }!!
    }

    fun decode(id: Int): String {
        return postprocess(reversedVocab.getOrDefault(id, ""))
    }

    /*
    Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split further.
     */
    private fun recursiveSplit(segment: String, isFinal: Boolean): List<String> {
        val result: MutableList<String> = ArrayList()
        val left: String
        val right: String
        if (isFinal) {
            if ("$segment</w>" !in reversedCodes) {
                return listOf(segment)
            }
            val pair = reversedCodes["$segment</w>"]!!
            left = pair.word1; right = pair.word2.substring(0, pair.word2.length - 4)
        } else {
            if (segment !in reversedCodes) {
                return listOf(segment)
            }
            val pair = reversedCodes[segment]!!
            left = pair.word1; right = pair.word2
        }
        if (left in vocab) {
            result.add(left)
        } else {
            result.addAll(recursiveSplit(left, isFinal))
        }
        if (isFinal && vocab.containsKey(right) || !isFinal && right in vocab) {
            result.add(right)
        } else {
            result.addAll(recursiveSplit(right, isFinal))
        }
        return result
    }

    /*
    Check for each segment in word if it is in-vocabulary,
    and segment OOV segments into smaller units by reversing the BPE merge operations"""
     */
    private fun checkVocabAndSplit(pieces: List<String>): MutableList<String> {
        val out = ArrayList<String>()
        for (i in 0 until pieces.size - 1) {
            val segment = pieces[i]
            if (segment in vocab) {
                out.add(segment)
            } else {
                out.addAll(recursiveSplit(segment, false))
            }
        }
        val segment = pieces[pieces.size - 1]
        if (segment in vocab) {
            out.add(segment)
        } else {
            out.addAll(recursiveSplit(segment, true))
        }
        return out
    }
}
