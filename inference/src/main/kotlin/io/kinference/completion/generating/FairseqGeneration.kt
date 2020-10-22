package io.kinference.completion.generating

import io.kinference.completion.BPETokenizer
import io.kinference.completion.GenerationConfig
import io.kinference.ndarray.*
import java.lang.Integer.min
import kotlin.math.ln

class FairseqGeneration(val model: ModelWrapper, private val tokenizer: BPETokenizer) {
    private val prefixMatcher = FuzzyPrefixMatcher(tokenizer)

    private var prefixes: List<Pair<String, Int>>? = null  // ArrayList()
    private var mems: List<MutableNDArray>? = null

    private val padTokenId = tokenizer.eosTokenId
    private val eosTokenId = tokenizer.eosTokenId
    private val vocabSize = tokenizer.vocabSize

    private var logSpellProb = ln(0.0001)
//    private val lenNormBase = 5.0
//    private val lenNormPow = 0.6
//    private val diversityStrength = 1

//    private val verbose = false


    private fun getSearch(config: GenerationConfig): Search {
        if (config.numGroups > 1) {
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
//            if (verbose) {
//                print("Using Beam search")
//                print("Search parameters: " +
//                    "vocab_size: $vocabSize, " +
//                    "beam_size: $numBeams, " +
//                    "len_norm_base: $lenNormBase, " +
//                    "len_norm_pow: $lenNormPow, " +
//                    "repetition_penalty: $repetitionPenalty"
//                )
//            }
            return BeamSearch(
                intArrayOf(eosTokenId),
                vocabSize,
                config.numBeams,
                config.lenNormBase,
                config.lenNormPow,
                config.repetitionPenalty
            )
        }
    }

    private fun modifyScore(scores: Array<DoubleArray>): Array<DoubleArray> {
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

    private fun initLogProbs(context: IntArray): Array<DoubleArray> {
        val logProbs = model.initLastLogProbs(arrayOf(context))
        val scores = logProbs.first
        mems = logProbs.second

        return modifyScore(logSoftmax(scores))
    }

    private fun initState(context: IntArray, prefix: String, config: GenerationConfig): Array<DoubleArray> {
        logSpellProb = ln(config.spellProb)
        prefixes = listOf(Pair(prefix, config.prefixErrLimit))
        return initLogProbs(context)
    }

    @ExperimentalUnsignedTypes
    private fun sortState(sortMask: IntArray) {
        // mems = [mem[:, sort_mask].contiguous() for mem in mems]
        mems = mems!!.map { mem ->
            val shape = mem.shape
            val outputStrides  = Strides(intArrayOf(shape[0], sortMask.size, shape[2], shape[3], shape[4]))
            val array = MutableFloatNDArray(FloatArray(outputStrides.linearSize), outputStrides)
            val rowLen = mem.linearSize / shape[0]
            val localRowLen = rowLen / shape[1]
            var off = 0
            for (i in 0 until shape[0]) {
                val rowStart = i * rowLen
                for (j in sortMask.indices) {
                    val totalRowOffset = rowStart + localRowLen * sortMask[j]
                    array.placeFrom(off, mem, totalRowOffset, totalRowOffset + localRowLen)
                    off += localRowLen
                }
            }
            array
        }

        prefixes = prefixes!!.slice(sortMask)
    }

    private fun getLogProbs(data: IntArray): Array<DoubleArray> {
        val logProbs = model.getLastLogProbs(data, mems!!)
        val scores = logProbs.first
        mems = logProbs.second
        return modifyScore(logSoftmax(scores))
    }

    private fun updatePrefix(newTokensIds: IntArray) {
        if (prefixes != null) {
            val result = ArrayList<Pair<String, Int>>(prefixes!!.size)

            prefixes!!.forEachIndexed { i, (prefix, errLimit) ->
                val tokenId = newTokensIds[i]
                val token = tokenizer.decode(tokenId)
                val errCnt = PrefixMatcher.levenshtein(prefix, token)
                val newPrefix = prefix.substring(min(prefix.length, token.length))
                result.add(Pair(newPrefix, min(errLimit - errCnt, newPrefix.length)))
            }

            prefixes = result
        }
    }

    private fun resetState() {
        mems = null
        prefixes = null
    }

    private fun isEndOfWords(trueScores: Array<DoubleArray>): BooleanArray {
        val endOfWords = BooleanArray(prefixes!!.size)
        val tokensIds = topk2d(trueScores, 3, dim = 1)  // both (batch_size * num_beams, 3)

        prefixes!!.forEachIndexed { batch_id, (prefix, err_limit) ->
            val tokenIds = tokensIds[batch_id]
            val tokens = tokenIds.map { tokenizer.decode(it) }
            val isEndOfWord = tokens.map { !it[0].isLetter() && PrefixMatcher.levenshtein(it, prefix) <= err_limit }.any()
            endOfWords[batch_id] = isEndOfWord
        }

        return endOfWords
    }

    fun generate(context: IntArray, prefix: String, config: GenerationConfig):
        List<Pair<List<List<Pair<IntArray, GenerationInfo>>>, List<List<Pair<IntArray, GenerationInfo>>>>> {
        val search = getSearch(config)

        val oneLogProbs = initState(context, prefix, config)
        var logProbs = Array(search.batchSize) { oneLogProbs[0] }
        sortState(IntArray(search.batchSize))

        val result = ArrayList<Pair<List<List<Pair<IntArray, GenerationInfo>>>, List<List<Pair<IntArray, GenerationInfo>>>>>()
        for (i in 0 until config.maxLen) {
            val selectedInds = search.step(logProbs, context)
            sortState(selectedInds)

            val lastPredictions = search.lastPredictions()
            updatePrefix(lastPredictions)

            logProbs = getLogProbs(lastPredictions)
            val isEndOfWords = isEndOfWords(logProbs)
            result.add(Pair(search.terminatedHypotheses(), search.maskedHypotheses(isEndOfWords)))
        }

        resetState()
        return result
    }
}
