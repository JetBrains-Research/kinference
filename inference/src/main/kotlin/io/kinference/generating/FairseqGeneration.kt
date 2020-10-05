package io.kinference.generating

import io.kinference.ndarray.*
import java.lang.Integer.min
import kotlin.math.ln

class FairseqGeneration(val model: ModelWrapper, private val tokenizer: BPETokenizer, private val prefixErrLimit: Int, spellProb: Double) {
    private val prefixMatcher = FuzzyPrefixMatcher(tokenizer, prefixErrLimit)

    private var prefixes: List<Pair<String, Int>>? = null  // ArrayList()
    private var mems: List<MutableNDArray>? = null

    private val padTokenId = tokenizer.eosTokenId
    private val eosTokenId = tokenizer.eosTokenId
    private val vocabSize = tokenizer.vocabSize

    private val logSpellProb = ln(spellProb)
    private val lenNormBase = 5.0
    private val lenNormPow = 0.6
    private val diversityStrength = 1

    private val verbose = false


    private fun getSearch(numBeams: Int, numGroups: Int = 1, repetitionPenalty: Double = 1.0): Search {
        if (numGroups > 1) {
            throw IllegalArgumentException("num groups > 1 is not supported")

//            if (verbose) {
//                print("\nUsing Diverse search")
//                print(
//                    f"Search parameters: "
//                    f "vocab_size: {self.vocab_size}, "
//                    f "beam_size: {num_beams}, "
//                    f "num_groups: {num_groups}, "
//                    f "diversity_strength: {self._diversity_strength}, "
//                    f "len_norm_base: {self._len_norm_base}, "
//                    f "len_norm_pow: {self._len_norm_pow}, "
//                    f "repetition_penalty: {repetition_penalty}"
//                )
//            }
//            return DiverseBeamSearch(
//                eos_ids = [self.eos_token_id],
//                vocab_size = self.vocab_size,
//                search_size = num_beams,
//                num_groups = num_groups,
//                diversity_strength = self._diversity_strength,
//                len_norm_base = self._len_norm_base,
//                len_norm_pow = self._len_norm_pow,
//                repetition_penalty = repetition_penalty
//            )
        } else {
            if (verbose) {
                print("Using Beam search")
                print("Search parameters: " +
                    "vocab_size: $vocabSize, " +
                    "beam_size: $numBeams, " +
                    "len_norm_base: $lenNormBase, " +
                    "len_norm_pow: $lenNormPow, " +
                    "repetition_penalty: $repetitionPenalty"
                )
            }
            return BeamSearch(
                intArrayOf(eosTokenId),
                vocabSize,
                numBeams,
                lenNormBase,
                lenNormPow,
                repetitionPenalty
            )
        }
    }

    private fun modifyScore(scores: List<MutableList<Double>>): List<MutableList<Double>> {
        // prefix

        prefixes!!.forEachIndexed { i, (prefix, err_limit) ->
            if (prefix != "") {
                val prefixIndsByErr = prefixMatcher.prefixTokensByErr(prefix, err_limit)
                for (j in prefixIndsByErr[0]) {
                    scores[i][j] = Double.NEGATIVE_INFINITY
                }

                for (err_num in 1 until prefixIndsByErr.size) {
                    val prefixToken = prefixIndsByErr[err_num]
                    if (err_num != 0) {
                        for (j in prefixToken) {
                            scores[i][j] = scores[i][j] + err_num * logSpellProb
                        }
                    }
                }
            }
        }

        return scores
    }

    private fun initLogProbs(context: List<Int>): List<MutableList<Double>> {
        val logProbs = model.initLastLogProbs(listOf(context))
        val scores = logProbs.first
        mems = logProbs.second

        return modifyScore(logSoftmax(scores))
    }

    private fun initState(context: List<Int>, prefix: String): List<MutableList<Double>> {
        prefixes = listOf(Pair(prefix, prefixErrLimit))
        return initLogProbs(context)
    }

    @ExperimentalUnsignedTypes
    private fun sortState(sortMask: List<Int>) {
        // mems = [mem[:, sort_mask].contiguous() for mem in mems]
        mems = mems!!.map { mem ->
            val shape = mem.shape
            val size = mem.linearSize * sortMask.size / shape[1]
            val values: MutableList<Float> = ArrayList(size)

            for (i in 0 until shape[0]) {
                val row = mem.row(i)
                for (j in sortMask.indices) {
                    values.addAll((row.row(sortMask[j]) as FloatNDArray).array.toList())
                }
            }
            MutableFloatNDArray(values.toFloatArray(), Strides(intArrayOf(shape[0], sortMask.size, shape[2], shape[3], shape[4])))
        }

        prefixes = prefixes!!.slice(sortMask)
    }

    private fun getLogProbs(data: List<Int>): List<MutableList<Double>> {
        val logProbs = model.getLastLogProbs(data, mems!!)
        val scores = logProbs.first
        mems = logProbs.second
        return modifyScore(logSoftmax(scores))
    }

    private fun updatePrefix(newTokensIds: List<Int>) {
        if (prefixes != null) {
            val result = ArrayList<Pair<String, Int>>(prefixes!!.size)

            prefixes!!.forEachIndexed { i, (prefix, errLimit) ->
                val tokenId = newTokensIds[i]
                val token = tokenizer.decode(tokenId)
                val errCnt = PrefixMatcher.levenshtein(prefix, token)
                val newPrefix = prefix.substring(token.length)
                result.add(Pair(newPrefix, min(errLimit - errCnt, newPrefix.length)))
            }

            prefixes = result
        }
    }

    private fun resetState() {
        mems = null
        prefixes = null
    }

    private fun isEndOfWords(trueScores: List<List<Double>>): List<Boolean> {
        val endOfWords = ArrayList<Boolean>()
        val tokensIds = topk2d(trueScores, 3, dim = 1)  // both (batch_size * num_beams, 3)

        prefixes!!.forEachIndexed { batch_id, (prefix, err_limit) ->
            val tokenIds = tokensIds[batch_id]
            val tokens = tokenIds.map { tokenizer.decode(it) }
            val isEndOfWord = tokens.map { !it[0].isLetter() && PrefixMatcher.levenshtein(it, prefix) <= err_limit }.any()
            endOfWords.add(isEndOfWord)
        }

        return endOfWords
    }

    fun generate(context: List<Int>, prefix: String, numBeams: Int, numIterations: Int, numGroups: Int = 1, repetitionPenalty: Double = 1.0):
        List<Pair<List<List<Pair<List<Int>, GenerationInfo>>>, List<List<Pair<List<Int>, GenerationInfo>>>>> {
        val search = getSearch(numBeams, numGroups, repetitionPenalty)

        val one_log_probs = initState(context, prefix)

        var log_probs = List(search.batchSize()) { one_log_probs[0] }
        sortState(List(search.batchSize()) { 0 })

        val result = ArrayList<Pair<List<List<Pair<List<Int>, GenerationInfo>>>, List<List<Pair<List<Int>, GenerationInfo>>>>>()
        for (i in 0 until numIterations) {
            val selected_inds = search.step(log_probs, context)
            sortState(selected_inds)

            val last_predictions = search.lastPredictions()
            updatePrefix(last_predictions)

            log_probs = getLogProbs(last_predictions)
            val is_end_of_words = isEndOfWords(log_probs)
            result.add(Pair(search.terminatedHypotheses(), search.maskedHypotheses(is_end_of_words)))
        }

        resetState()
        return result
    }
}
