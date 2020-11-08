package io.kinference.algorithms.completion

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import java.io.File
import java.io.IOException

class BPETokenizer(vocabPath: String, mergesPath: String) {
    private data class MergeCandidate(val word1: String, val word2: String) {
        fun merged(): String = word1 + word2
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
        val mapType = object : TypeReference<HashMap<String, Int>>() {}
        val vocabJson = ObjectMapper().readValue(File(vocabPath), mapType)
        for (pair in vocabJson.entries) {
            val value = pair.value
            vocab[pair.key] = value
            reversedVocab[value] = pair.key
        }

        var merges = File(mergesPath).readLines()
        if (merges[0][0] == '#') {
            merges = merges.drop(1)
        }
        for (line in merges) {
            val words = line.split(" ")
            val pair = MergeCandidate(words[0], words[1])
            codes[pair] = codes.size
            reversedCodes[words[0] + words[1]] = pair
        }
    }

    private fun preprocess(s: String): String {
        return s.replace(' ', 'Ġ')
    }

    private fun postprocess(s: String): String {
        return s.replace('Ġ', ' ')
    }

    private fun internalTokenize(word: String): List<String> {
        if (word.length == 1) {
            return listOf(word)
        }
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
            val strBigram = bigram.merged()
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

    fun tokenize(s: String): List<String> {
        val tokens = internalTokenize(preprocess(s))
        return tokens.map { postprocess(it) }
    }

    fun encode(s: String): IntArray {
        val tokens = internalTokenize(preprocess(s))
        return IntArray(tokens.size) { vocab.getOrDefault(tokens[it], 0) }
    }

    fun decode(ids: IntArray): String {
        return postprocess(ids.joinToString("") { reversedVocab.getOrDefault(it, "") })
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

    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            val vocabPath = "/Users/aleksandr.khvorov/.cache/torch/transformers/71cc2431cf0b5bbe7a23601a808ed322c90251c8261b46f04970140a3c2c1cb4.1512018be4ba4e8726e41b9145129dc30651ea4fec86aa61f4b9f40bf94eac71"
            val mergesPath = "/Users/aleksandr.khvorov/.cache/torch/transformers/4faf7afb02a1ea7d2944e9ba7a175c7b8de4957cdbae75cd5ddffc7c7643ebbc.70bec105b4158ed9a1747fea67a43f5dee97855c64d62b6ec3742f4cfdb5feda"

//            val vocabPath = "/Users/aleksandr.khvorov/jb/other/vocab.json"
//            val mergesPath = "/Users/aleksandr.khvorov/jb/other/merges.json"
            try {
                val tokenizer = BPETokenizer(vocabPath, mergesPath)
                val text = "1. Modeling Vocabulary for Big Code Machine Learning (https://arxiv.org/pdf/1904.01873.pdf)"
                val res = tokenizer.tokenize(text)
                print(res.map { "'$it'" })
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
    }
}
