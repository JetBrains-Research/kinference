package io.kinference.algorithms.completion.generation

import io.kinference.algorithms.completion.CompletionConfig
import io.kinference.algorithms.completion.generation.matcher.FuzzyPrefixMatcher
import io.kinference.algorithms.completion.generation.matcher.PrefixMatcher
import io.kinference.algorithms.completion.generation.model.ModelWrapper
import io.kinference.algorithms.completion.generation.search.BeamSearch
import io.kinference.algorithms.completion.generation.search.Search
import io.kinference.algorithms.completion.tokenizer.BPETokenizer
import io.kinference.ndarray.arrays.MutableFloatNDArray
import io.kinference.ndarray.arrays.NDArray
import kotlin.math.*

internal class FairSeqGeneration(private val model: ModelWrapper, private val tokenizer: BPETokenizer) {
    data class PrefixInfo(val text: String, val errLimit: Int)

    private val prefixMatcher = FuzzyPrefixMatcher(tokenizer)

    private var prefixes: List<PrefixInfo>? = null
    private var mems: List<NDArray>? = null
    private var eachStepProbs: List<MutableList<Double>> = listOf(ArrayList())
    private var nextLogProbs: Array<DoubleArray>? = null

    private val vocabSize: Int
        get() = tokenizer.vocabSize

    private var logSpellProb = ln(0.0001)

    private fun getSearch(config: CompletionConfig.Generation): Search {
        require(config.numGroups == 1) { "num groups > 1 is not supported" }

        return BeamSearch(vocabSize, config.numBeams, config.repetitionPenalty)
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

            // ban tokens with bad symbols
//            for (j in tokenizer.invalidIds) {
//                scores[i][j] = Double.NEGATIVE_INFINITY
//            }
        }

        return scores
    }

    private fun initLogProbs(context: IntArray) {
        val logProbs = model.initLastLogProbs(arrayOf(context))
        mems = logProbs.pastStates
        nextLogProbs = modifyScore(logSoftmax(logProbs.logProbs))
    }

    private fun initState(prefix: String, config: CompletionConfig.Generation) {
        logSpellProb = ln(config.spellProb)
        prefixes = listOf(PrefixInfo(prefix, config.prefixErrLimit))
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

        eachStepProbs = sortMask.map { ArrayList(eachStepProbs[it]) }
        prefixes = prefixes!!.slice(sortMask)
    }

    private fun updateState(sortMask: IntArray, newTokensIds: IntArray) {
        sortState(sortMask)

        sortMask.zip(newTokensIds).forEachIndexed {
            index, (batchInd, tokenInd) -> eachStepProbs[index].add(exp(nextLogProbs!![batchInd][tokenInd]))
        }

        updatePrefix(newTokensIds)
    }

    private fun getLogProbs(data: IntArray) {
        val logProbs = model.getLastLogProbs(data, mems!!)
        mems = logProbs.pastStates

        nextLogProbs = modifyScore(logSoftmax(logProbs.logProbs))
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

    private fun currentHypothesis(search: Search): List<GenerationInfo> {
        return search.hypotheses().zip(eachStepProbs).map { (hyp, probs) -> GenerationInfo(probs, hyp) }
    }

    fun generate(context: IntArray, prefix: String, config: CompletionConfig.Generation): List<List<GenerationInfo>> {
        val search = getSearch(config)

        initState(prefix, config)
        initLogProbs(context)
        sortState(IntArray(search.batchSize))

        val result = ArrayList<List<GenerationInfo>>()
        for (i in 0 until config.maxLen) {
            val stepResult = search.step(nextLogProbs!!, context)
            updateState(stepResult.sortMask, stepResult.newTokens)

            if (i < config.maxLen - 1) {
                getLogProbs(stepResult.newTokens)
            }
            result.add(currentHypothesis(search))
        }

        resetState()
        return result
    }
}
