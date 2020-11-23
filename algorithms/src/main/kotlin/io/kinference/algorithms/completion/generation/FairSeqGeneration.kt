package io.kinference.algorithms.completion.generation

import io.kinference.algorithms.completion.config.GenerationConfig
import io.kinference.algorithms.completion.generation.matcher.FuzzyPrefixMatcher
import io.kinference.algorithms.completion.generation.matcher.PrefixMatcher
import io.kinference.algorithms.completion.generation.model.ModelWrapper
import io.kinference.algorithms.completion.generation.search.BeamSearch
import io.kinference.algorithms.completion.generation.search.Search
import io.kinference.algorithms.completion.tokenizer.BPETokenizer
import io.kinference.ndarray.arrays.MutableFloatNDArray
import io.kinference.ndarray.arrays.NDArray
import kotlin.math.ln
import kotlin.math.min

class FairSeqGeneration(private val model: ModelWrapper, private val tokenizer: BPETokenizer) {
    data class PrefixInfo(val text: String, val errLimit: Int)

    private val prefixMatcher = FuzzyPrefixMatcher(tokenizer)

    private var prefixes: List<PrefixInfo>? = null
    private var mems: List<NDArray>? = null

    private val eosTokenId: Int
        get() = tokenizer.eosTokenId

    private val vocabSize: Int
        get() = tokenizer.vocabSize

    private var logSpellProb = ln(0.0001)

    private fun getSearch(config: GenerationConfig): Search {
        require(config.numGroups == 1) { "num groups > 1 is not supported" }

        return BeamSearch(
            intArrayOf(eosTokenId),
            vocabSize,
            config.numBeams,
            config.lenNormBase,
            config.lenNormPow,
            config.repetitionPenalty
        )
    }

    private fun modifyScore(scores: Array<DoubleArray>): Array<DoubleArray> {
        prefixes!!.forEachIndexed { i, (prefix, err_limit) ->
            if (prefix.isEmpty()) return@forEachIndexed

            val prefixIndsByErr = prefixMatcher.prefixTokensByErr(prefix, err_limit)
            for (j in prefixIndsByErr[0]) {
                scores[i][j] = Double.NEGATIVE_INFINITY
            }

            for (err_num in 1 until prefixIndsByErr.size) {
                val prefixToken = prefixIndsByErr[err_num]
                for (j in prefixToken) {
                    scores[i][j] = scores[i][j] + err_num * logSpellProb
                }
            }
        }

        return scores
    }

    private fun initLogProbs(context: IntArray): Array<DoubleArray> {
        val logProbs = model.initLastLogProbs(arrayOf(context))
        mems = logProbs.pastStates

        return modifyScore(logSoftmax(logProbs.logProbs))
    }

    private fun initState(context: IntArray, prefix: String, config: GenerationConfig): Array<DoubleArray> {
        logSpellProb = ln(config.spellProb)
        prefixes = listOf(PrefixInfo(prefix, config.prefixErrLimit))
        return initLogProbs(context)
    }

    private fun sortState(sortMask: IntArray) {
        // mems = [mem[:, sort_mask].contiguous() for mem in mems]
        mems = mems!!.map { mem ->
            val shape = mem.shape
            val outputShape = intArrayOf(shape[0], sortMask.size, shape[2], shape[3], shape[4])
            val array = MutableFloatNDArray(shape = outputShape)
            val rowLen = mem.linearSize / shape[0]
            val localRowLen = rowLen / shape[1]
            var off = 0
            for (i in 0 until shape[0]) {
                val rowStart = i * rowLen
                for (j in sortMask.indices) {
                    val totalRowOffset = rowStart + localRowLen * sortMask[j]
                    array.copyFrom(off, mem, totalRowOffset, totalRowOffset + localRowLen)
                    off += localRowLen
                }
            }
            array
        }

        prefixes = prefixes!!.slice(sortMask)
    }

    private fun getLogProbs(data: IntArray): Array<DoubleArray> {
        val logProbs = model.getLastLogProbs(data, mems!!)
        mems = logProbs.pastStates

        return modifyScore(logSoftmax(logProbs.logProbs))
    }

    private fun updatePrefix(newTokensIds: IntArray) {
        if (prefixes == null) return

        val result = ArrayList<PrefixInfo>(prefixes!!.size)

        prefixes!!.forEachIndexed { i, (prefix, errLimit) ->
            val tokenId = newTokensIds[i]
            val token = tokenizer.decode(tokenId)
            val errCnt = PrefixMatcher.levenshtein(prefix, token)
            val newPrefix = prefix.substring(min(prefix.length, token.length))
            result.add(PrefixInfo(newPrefix, min(errLimit - errCnt, newPrefix.length)))
        }

        prefixes = result
    }

    private fun resetState() {
        mems = null
        prefixes = null
    }

    fun generate(context: IntArray, prefix: String, config: GenerationConfig): List<List<List<Search.HypothesisInfo>>> {
        val search = getSearch(config)

        val oneLogProbs = initState(context, prefix, config)
        var logProbs = Array(search.batchSize) { oneLogProbs[0] }
        sortState(IntArray(search.batchSize))

        val result = ArrayList<List<List<Search.HypothesisInfo>>>()
        for (i in 0 until config.maxLen) {
            val selectedInds = search.step(logProbs, context)
            sortState(selectedInds)

            val lastPredictions = search.lastPredictions()
            updatePrefix(lastPredictions)

            if (i < config.maxLen - 1) {
                logProbs = getLogProbs(lastPredictions)
            }
            result.add(search.currentHypotheses())
        }

        resetState()
        return result
    }
}
