package io.kinference.ndarray.generating

import com.beust.klaxon.Parser
import com.beust.klaxon.JsonObject
import java.io.File
import java.io.IOException
import java.util.*
import kotlin.collections.HashMap


class BPETokenizer(vocabPath: String, mergesPath: String) {
    private val vocab: MutableMap<String, Int> = HashMap()
    private val reversedVocab: MutableMap<Int, String> = HashMap()
    private val codes: MutableMap<Pair<String, String>, Int> = HashMap()
    private val reversedCodes: MutableMap<String, Pair<String, String>> = HashMap()

    val eosTokenId = 50256
    val vocabSize: Int
        get() {
            return vocab.size
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
        var pieces: MutableList<String> = ArrayList()
        for (element in word) {
            pieces.add("" + element)
        }

        while (pieces.size > 1) {
            val pairs: MutableList<Triple<Int?, Int, Pair<String, String>>> = ArrayList(pieces.size)
            for (i in 0 until pieces.size - 1) {
                val current = pieces[i]
                val next = pieces[i + 1]
                val pair = Pair(current, next)
                if (pair in codes) {
                    pairs.add(Triple(codes[pair], i, pair))
                }
            }
            if (pairs.isEmpty()) {
                break
            }

            pairs.sortWith(compareBy({ it.first }, { it.second }))
            val bigram = pairs[0].third
            val positions = pairs.filter { it.third == bigram }.map { it.second }

            var i = 0
            val newPieces: MutableList<String> = ArrayList()
            val strBigram = java.lang.String.join("", bigram.component1(), bigram.component2())
            for (j in positions) {
                if (j < i) {
                    continue
                }
                newPieces.addAll(pieces.subList(i, j))
                newPieces.add(strBigram)
                i = j + 2
            }
            newPieces.addAll(pieces.subList(i, pieces.size))
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

    fun encode(s: String): List<Int> {
        val tokens = internalTokenize(preprocess(s))
        return tokens.map { vocab.getOrDefault(it, 0) }
    }

    fun decode(ids: List<Int>): String {
        return preprocess(ids.map { reversedVocab.getOrDefault(it, "") }.joinToString { "" })
    }

    fun decode(id: Int): String {
        return preprocess(reversedVocab.getOrDefault(id, ""))
    }

    /*
    Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split futher.
     */
    private fun recursiveSplit(segment: String, isFinal: Boolean): List<String> {
        val result: MutableList<String> = ArrayList()
        val left: String
        var right: String
        if (isFinal) {
            if ("$segment</w>" !in reversedCodes) {
                return listOf(segment)
            }
            val pair = reversedCodes["$segment</w>"]!!
            left = pair.component1()
            right = pair.component2()
            right = right.substring(0, right.length - 4)
        } else {
            if (segment !in reversedCodes) {
                return listOf(segment)
            }
            val pair = reversedCodes[segment]!!
            left = pair.component1()
            right = pair.component2()
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
        val out: MutableList<String> = ArrayList()
        for (segment in pieces.subList(0, pieces.size - 1)) {
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

    init {
        val parser: Parser = Parser.default()
        val vocabJson: JsonObject = parser.parse(vocabPath) as JsonObject
        for (pair in vocabJson.map.entries) {
            val value = pair.value as Int
            vocab[pair.key] = value
            reversedVocab[value] = pair.key
        }

//        var vocabLine = File(vocabPath).readLines()[0]
//        vocabLine = vocabLine.substring(1, vocabLine.length - 2)
//        val pieces = vocabLine.split("\"")
//        var i = 1
//        var lastInd = 1
//        var key: String?
//        var value: Int?
//        val g = 'Ġ'
//        while (i < pieces.size) {
//            val s = pieces[i]
//            if (s.endsWith("Model")) {
//                var a = 0
//            }
//            if (s.matches(Regex(": [0-9]+, "))) {
//                value = s.substring(2, s.indexOf(',')).toInt()
//                key = pieces.subList(lastInd, i).joinToString("\"")
//                lastInd = i + 1
//
//                vocab[key] = value
//                reversedVocab[value] = key
//            }
//            i += 1
//            if (i % 100 == 0) {
//                var a = 0
//            }
//        }

        var merges = File(mergesPath).readLines()
//        val merges: JsonArray<String> = parser.parse(mergesPath) as JsonArray<String>
        if (merges[0][0] == '#') {
            merges = merges.subList(1, merges.size)
        }
        for (line in merges) {
            val words = line.split(" ")
            val pair = Pair(words[0], words[1])
            codes[pair] = codes.size
            reversedCodes[words[0] + words[1]] = pair
        }
    }
}
