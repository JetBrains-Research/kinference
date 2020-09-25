package io.kinference.ndarray.generating

import io.kinference.ndarray.*
import io.kinference.primitives.types.DataType
import java.lang.Math.*
import kotlin.math.min
import kotlin.math.pow

abstract class Search(val eosIds: IntArray, val vocabSize: Int, val searchSize: Int,
                      val lenNormBase: Int = 0, val lenNormPow: Int = 0, val repetitionPenalty: Double = 1.0) {

    @ExperimentalUnsignedTypes
    abstract fun step(stepLogProbs: MutableNumberNDArray, context: IntNDArray? = null): List<Int>

    protected fun stepCheck(logProbs: NDArray) {
        assert(logProbs.shape.contentEquals(intArrayOf(batchSize(), vocabSize))
        ) { "log_probs must have shape (${batchSize()}, $vocabSize), but ${logProbs.shape} was given" }

        assert(eosIds.all { it < vocabSize }
        ) { "EOS ids must be less than vocab_size, but EOS ids: $eosIds and vocab_size: $vocabSize" }
    }

    /**
     * List of list of tuples of current hypotheses and theirs scores
     */
    abstract fun maskedHypotheses(mask: BooleanArray): List<List<Pair<List<Int>, GenerationInfo>>>

    /**
     * List of list of tuples of terminated hypotheses and theirs scores
     */
    abstract fun terminatedHypotheses(): List<List<Pair<List<Int>, GenerationInfo>>>

    /**
     * List of list of tuples of current hypotheses and theirs scores
     */
    abstract fun currentHypotheses(): List<List<Pair<List<Int>, GenerationInfo>>>

    /**
     * Tensor of last tokens of the current hypotheses with shape (batch_size,) to make a batch for a model
     */
    abstract fun lastPredictions(): NDArray

    /**
     * Current batch size
     */
    abstract fun batchSize(): Int
}

class BeamSearch(eosIds: IntArray, vocabSize: Int, searchSize: Int,
                 lenNormBase: Int = 0, lenNormPow: Int = 0, repetitionPenalty: Double = 1.0) :
    Search(eosIds, vocabSize, searchSize, lenNormBase, lenNormPow, repetitionPenalty) {

    val length = 1.0

    //    private var scores: MutableNumberNDArray? = null
    private var scores: MutableList<Float> = ArrayList()
    private var hypotheses: List<MutableList<Int>> = ArrayList()
    private var eachStepProbs: List<MutableList<Float>> = ArrayList()
    private val terminatedHypotheses: MutableList<Pair<List<Int>, Float>> = ArrayList()
    private var sortMask: List<Int>? = null
    private var eosTensor: NDArray? = null

    private fun initState(type: DataType) {
//        scores = torch.zeros(1, dtype = type)
//        hypotheses = torch.empty(1, 0, dtype = torch.long)
//        each_step_probs = torch.empty(1, 0, dtype = dtype)
//        eos_tensor = torch.tensor(self._eos_ids, dtype = torch.long).unsqueeze(1)
    }

    @ExperimentalUnsignedTypes
    override fun step(stepLogProbs: MutableNumberNDArray, context: IntNDArray?): List<Int> {
        modifyScore(stepLogProbs, context)
        super.stepCheck(stepLogProbs)

        if (scores == null) {
            initState(stepLogProbs.type)
        }

//        var logProbs = stepLogProbs.plus(scores!!.unsqueeze(1) as MutableNumberNDArray)
        var logProbs = stepLogProbs.plus(scores as MutableNumberNDArray)
        logProbs = logProbs.reshape(intArrayOf(logProbs.linearSize))

        var stepLogProbs = stepLogProbs.map(object : DoubleMap {
            override fun apply(value: Double): Double = kotlin.math.exp(value)
        }).reshape(intArrayOf(stepLogProbs.linearSize))

        var topInds = topk1d(logProbs, min((1 + eosIds.size) * searchSize, logProbs.shape[0]), sorted = false)
        val samplesStepLogProbs = topInds.map { stepLogProbs[it] }
        val sortMask = topInds.map { floorDiv(it, vocabSize) }
        topInds = topInds.map { floorMod(it, vocabSize) }

        initSortMask()
        val sampleScores = topInds.map { logProbs[it] }
        updateState(topInds, sampleScores, samplesStepLogProbs, sortMask)
        length.plus(1)

        return sortMask
    }

    private fun topk(data: NumberNDArray, size: Int, sorted: Boolean = true): List<List<Int>> {
        return emptyList()
    }

    private fun topk1d(data: NumberNDArray, size: Int, sorted: Boolean = true): List<Int> {
        assert(data.shape.size == 1)
        return listOf()
    }

    private fun topk1dList(data: List<Float>, size: Int): List<Int> {
        val pairedData = ArrayList<Pair<Float, Int>>()
        for (i in data.indices) {
            pairedData.add(Pair(data[i], i))
        }
        pairedData.sortBy { -it.first }
        return pairedData.map { it.second }.subList(0, size)
    }

    @ExperimentalUnsignedTypes
    private fun modifyScore(scores: MutableNumberNDArray, context: IntNDArray?) {
        // repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)

        if (repetitionPenalty != 1.0) {
            if (context != null) {
                val contextTokens = context.row(0) as IntNDArray
                val uniqueTokens = HashSet<Int>()
                for (i in 0 until contextTokens.shape[0]) {
                    uniqueTokens.add(contextTokens[i])
                }

                for (i in 0 until scores.shape[0]) {
                    pessimizeScore(scores, i, uniqueTokens)
                }
            }

            for (i in hypotheses.indices) {
                pessimizeScore(scores, i, hypotheses[i].toHashSet())
            }
        }
    }

    private fun pessimizeScore(scores: MutableNumberNDArray, ind: Int, uniqueTokens: Set<Int>) {
        val row = scores[ind] as MutableNumberNDArray

        for (previousToken in uniqueTokens) {
            val score = row[previousToken] as Double
            row[previousToken] = score * if (score < 0.0) repetitionPenalty else 1.0 / repetitionPenalty
        }

        scores[ind] = row
    }

    override fun maskedHypotheses(mask: BooleanArray): List<List<Pair<List<Int>, GenerationInfo>>> {
        val ans = ArrayList<Pair<List<Int>, GenerationInfo>>()
        for (i in hypotheses.indices) {
            if (mask[i]) {
//                val score = torch.exp(self._get_normalized_scores())
                val score = 0.0F
                ans.add(Pair(hypotheses[i], GenerationInfo(eachStepProbs[i], score)))
            }
        }
//        ans.sortBy { -it.second.score }

        return listOf(ans)
    }

    override fun terminatedHypotheses(): List<List<Pair<List<Int>, GenerationInfo>>> {
        val ans = ArrayList<Pair<List<Int>, GenerationInfo>>()
        for (hypothesis in terminatedHypotheses) {
            ans.add(Pair(hypothesis.first, GenerationInfo(score = hypothesis.second)))
        }

        // ans.sortBy { -it.second.score }
        return listOf(ans)
    }

    override fun currentHypotheses(): List<List<Pair<List<Int>, GenerationInfo>>> {
        val ans = ArrayList<Pair<List<Int>, GenerationInfo>>()
        for (i in hypotheses.indices) {
            // val score = torch.exp(self._get_normalized_scores())
            val score = 0.0F
            ans.add(Pair(hypotheses[i], GenerationInfo(eachStepProbs[i], score)))
        }

        // ans.sortBy { -it.second.score }
        return listOf(ans)
    }

    override fun lastPredictions(): NDArray {
        TODO("Not yet implemented")
    }

    override fun batchSize(): Int {
//        return scores?.shape?.get(0) ?: 1
        return scores.size
    }

    private fun initSortMask() {
        sortMask = (0 until batchSize()).toList()
    }

    private fun updateState(samples: List<Int>, sampleScores: List<Float>, stepLogProbs: NDArray, sortMask: List<Int>) {
        sortState(sortMask)

//        scores = sampleScores
//        self._hypotheses = torch.cat((self._hypotheses, samples.unsqueeze(1)), dim = 1)
//        eachStepProbs = torch.cat((eachStepProbs, stepLogProbs.unsqueeze(1)), dim = 1)
        stashTerminated(samples)
    }

    private fun stashTerminated(samples: List<Int>) {
        val toStash = isSampleTerminates(samples.subList(0, searchSize))

        scores = getNormalizedScores().map { kotlin.math.exp(it.toDouble()).toFloat() }.toMutableList()
        for (i in hypotheses.indices) {
            assert(hypotheses[i].size == length.toInt())
            hypotheses[i]
            terminatedHypotheses.add(Pair(ArrayList(hypotheses[i]), scores.subList(0, searchSize)[toStash])
        }

        for (terminatedHypothesis, score in zip(
        self._hypotheses[: self._search_size][to_stash], scores[: self._search_size][to_stash]
        ):
        assert len(terminatedHypothesis) == int(self._length)
        terminatedHypotheses.append((terminated_hypothesis.clone(), score.item()))

        val terminated = isSampleTerminates(samples)
//        applySliceToState(~terminated)
        sortState()
}

    private fun sortState(sortMask: List<Int>? = null) {
        val mask = topk1dList(scores, min(searchSize, scores.size))
        applySliceToState(mask)
    }

    private fun isSampleTerminates(samples: List<Int>): List<Boolean> {
//        val result = samples == eosTensor.expand(eosTensor!!.shape[0], samples.shape[0])
        //        return result.sum(dim = 0)

        val res = ArrayList<Boolean>()
        for (i in samples.indices) {
            res.add(false)
        }
        return res
    }

    private fun applySliceToState(tensorSlice: List<Int>) {
        scores = tensorSlice.map { scores[it] }.toMutableList()
        hypotheses = tensorSlice.map { hypotheses[it] }
        eachStepProbs = tensorSlice.map { eachStepProbs[it] }
        if (sortMask != null) {
            sortMask = tensorSlice.map { sortMask!![it] }
        }
}

    private fun getNormalizedScores(): List<Float> {
        val normFactor = ((lenNormBase + length) / (lenNormBase + 1)).pow(lenNormPow).toFloat()
//        return scores.div(normFactor)
        return scores.map { it / normFactor }
    }
}
