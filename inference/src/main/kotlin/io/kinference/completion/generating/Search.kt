package io.kinference.completion.generating

import io.kinference.ndarray.*
import io.kinference.primitives.types.DataType
import java.lang.Math.*
import kotlin.math.min
import kotlin.math.pow

abstract class Search(val eosIds: IntArray, val vocabSize: Int, val searchSize: Int,
                      val lenNormBase: Double = 0.0, val lenNormPow: Double = 0.0, val repetitionPenalty: Double = 1.0) {

    /**
     * Current batch size
     */
    abstract val batchSize: Int

    @ExperimentalUnsignedTypes
    abstract fun ndStep(ndStepLogProbs: MutableNumberNDArray, context: IntArray): IntArray

    abstract fun step(stepLogProbs: Array<DoubleArray>, context: IntArray): IntArray

    protected fun stepCheck(logProbs: NDArray) {
        assert(logProbs.shape.contentEquals(intArrayOf(batchSize, vocabSize))
        ) { "log_probs must have shape (${batchSize}, $vocabSize), but ${logProbs.shape} was given" }

        assert(eosIds.all { it < vocabSize }
        ) { "EOS ids must be less than vocab_size, but EOS ids: $eosIds and vocab_size: $vocabSize" }
    }

    /**
     * List of list of tuples of current hypotheses and theirs scores
     */
    abstract fun maskedHypotheses(mask: BooleanArray): List<List<Pair<IntArray, GenerationInfo>>>

    /**
     * List of list of tuples of terminated hypotheses and theirs scores
     */
    abstract fun terminatedHypotheses(): List<List<Pair<IntArray, GenerationInfo>>>

    /**
     * List of list of tuples of current hypotheses and theirs scores
     */
    abstract fun currentHypotheses(): List<List<Pair<IntArray, GenerationInfo>>>

    /**
     * Tensor of last tokens of the current hypotheses with shape (batch_size,) to make a batch for a model
     */
    abstract fun lastPredictions(): IntArray
}

class BeamSearch(eosIds: IntArray, vocabSize: Int, searchSize: Int,
                 lenNormBase: Double = 0.0, lenNormPow: Double = 0.0, repetitionPenalty: Double = 1.0) :
    Search(eosIds, vocabSize, searchSize, lenNormBase, lenNormPow, repetitionPenalty) {

    private var length = 1.0
    override val batchSize: Int
        get() = scores.size
    var scores: MutableList<Double> = arrayListOf(0.0)
        private set

    private var hypotheses: List<MutableList<Int>> = arrayListOf(arrayListOf())
    private var eachStepProbs: List<MutableList<Double>> = arrayListOf(arrayListOf())
    private val terminatedHypotheses: MutableList<Pair<List<Int>, Double>> = ArrayList()
    private var sortMask: IntArray? = null
    private val eosIdsSet: Set<Int> = eosIds.toSet()

    private fun initState(type: DataType) {
//        scores = torch.zeros(1, dtype = type)
//        hypotheses = torch.empty(1, 0, dtype = torch.long)
//        each_step_probs = torch.empty(1, 0, dtype = dtype)
//        eos_tensor = torch.tensor(self._eos_ids, dtype = torch.long).unsqueeze(1)
    }

    private fun toDoubleList2d(data: NumberNDArray): Array<DoubleArray> {
        assert(data.shape.size == 2)

        val rowLength: Int = data.linearSize / data.shape[0]
        return Array(data.shape[0]) {
            val rowStart = it * rowLength
            data.copyOfRange(rowStart, rowLength + rowLength) as DoubleArray
        }
    }

    @ExperimentalUnsignedTypes
    override fun ndStep(ndStepLogProbs: MutableNumberNDArray, context: IntArray): IntArray {
        ndModifyScore(ndStepLogProbs, context)
        super.stepCheck(ndStepLogProbs)

//        if (scores == null) {
//            initState(stepLogProbs.type)
//        }

        val stepLogProbs = toDoubleList2d(ndStepLogProbs)

        val ndScores = DoubleNDArray(scores.toDoubleArray(), Strides(intArrayOf(scores.size)))
        val ndLogProbs = ndStepLogProbs.plus(ndScores) as DoubleNDArray
        val logProbs = ndLogProbs.array

        //val flattenStepLogProbs = stepLogProbs.map { it.map { e -> kotlin.math.exp(e) }.toMutableList() }.flatten()
        val stepLogProbsLinearSize = stepLogProbs.sumBy { it.size }
        var offset = 0
        val expStepLogProbs = DoubleArray(stepLogProbsLinearSize)
        for (probs in stepLogProbs) {
            for (value in probs) expStepLogProbs[offset++] = kotlin.math.exp(value)
        }

        var samples = topk1d(logProbs, min((1 + eosIds.size) * searchSize, ndLogProbs.shape[0]))

        val samplesStepLogProbs = expStepLogProbs.sliceArray(samples)
        val sortMask = IntArray(samples.size) { floorDiv(samples[it], vocabSize) }
        samples = IntArray(samples.size) { floorMod(samples[it], vocabSize) }

        initSortMask()
        val sampleScores = logProbs.sliceArray(samples)
        updateState(samples, sampleScores, samplesStepLogProbs, sortMask)
        length.plus(1)

        return sortMask
    }

    override fun step(stepLogProbs: Array<DoubleArray>, context: IntArray): IntArray {
        modifyScore(stepLogProbs, context)
        // TODO: check
//        super.stepCheck(stepLogProbs)

//        if (scores == null) {
//            initState(stepLogProbs.type)
//        }
        val stepLogProbsLinearSize = stepLogProbs.sumBy { it.size }
        val logProbs = DoubleArray(stepLogProbsLinearSize)
        val expStepLogProbs = DoubleArray(stepLogProbsLinearSize)
        var offset = 0
        for (i in stepLogProbs.indices) {
            val probs = stepLogProbs[i]
            val score = scores[i]
            for (value in probs) {
                val currentVal = value + score
                logProbs[offset] = currentVal
                expStepLogProbs[offset++] = kotlin.math.exp(currentVal)
            }
        }

        var samples = topk1d(logProbs, min((1 + eosIds.size) * searchSize, logProbs.size))
        val sampleScores = logProbs.sliceArray(samples)

        val samplesStepLogProbs = expStepLogProbs.sliceArray(samples)
        val stepSortMask = IntArray(samples.size) { floorDiv(samples[it], vocabSize) }
        samples = IntArray(samples.size) { floorMod(samples[it], vocabSize) }

        initSortMask()
        updateState(samples, sampleScores, samplesStepLogProbs, stepSortMask)
        length += 1

        return sortMask!!
    }

    @ExperimentalUnsignedTypes
    private fun ndModifyScore(scores: MutableNumberNDArray, context: IntArray) {
        // repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)

        if (repetitionPenalty != 1.0) {
            for (i in 0 until scores.shape[0]) {
                ndPessimizeScore(scores, i, context.toSet())
            }

            for (i in hypotheses.indices) {
                ndPessimizeScore(scores, i, hypotheses[i].toSet())
            }
        }
    }

    private fun modifyScore(scores: Array<DoubleArray>, context: IntArray) {
        // repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)

        if (repetitionPenalty != 1.0) {
            val uniqueTokens = context.toSet()
            for (i in scores.indices) {
                pessimizeScore(scores, i, uniqueTokens)
            }

            for (i in hypotheses.indices) {
                pessimizeScore(scores, i, hypotheses[i].toSet())
            }
        }
    }

    private fun ndPessimizeScore(scores: MutableNumberNDArray, ind: Int, uniqueTokens: Set<Int>) {
        val row = scores[ind] as MutableNumberNDArray

        for (previousToken in uniqueTokens) {
            val score = row[previousToken] as Double
            row[previousToken] = score * if (score < 0.0) repetitionPenalty else 1.0 / repetitionPenalty
        }

        scores[ind] = row
    }

    private fun pessimizeScore(scores: Array<DoubleArray>, ind: Int, uniqueTokens: Set<Int>) {
        for (previousToken in uniqueTokens) {
            val score = scores[ind][previousToken]
            scores[ind][previousToken] = score * if (score < 0.0) repetitionPenalty else 1.0 / repetitionPenalty
        }
    }

    override fun maskedHypotheses(mask: BooleanArray): List<List<Pair<IntArray, GenerationInfo>>> {
        val ans = ArrayList<Pair<IntArray, GenerationInfo>>()
        val score = getNormalizedScores().map { kotlin.math.exp(it) }
        for (i in hypotheses.indices) {
            if (mask[i]) {
                ans.add(Pair(hypotheses[i].toIntArray(), GenerationInfo(eachStepProbs[i], score[i])))
            }
        }
//        ans.sortBy { -it.second.score }

        return listOf(ans)
    }

    override fun terminatedHypotheses(): List<List<Pair<IntArray, GenerationInfo>>> {
        val ans = ArrayList<Pair<IntArray, GenerationInfo>>()
        for (hypothesis in terminatedHypotheses) {
            ans.add(Pair(hypothesis.first.toIntArray(), GenerationInfo(score = hypothesis.second)))
        }

        // ans.sortBy { -it.second.score }
        return listOf(ans)
    }

    override fun currentHypotheses(): List<List<Pair<IntArray, GenerationInfo>>> {
        val score = getNormalizedScores().apply { for (i in this.indices) this[i] = kotlin.math.exp(this[i]) }
        val ans = List(hypotheses.size) {
            Pair(hypotheses[it].toIntArray(), GenerationInfo(eachStepProbs[it], score[it]))
        }

        // ans.sortBy { -it.second.score }
        return listOf(ans)
    }

    override fun lastPredictions(): IntArray {
        assert(hypotheses.isNotEmpty() && hypotheses[0].size > 0) {"Can't get last predictions if no steps have been performed"}
        return IntArray(hypotheses.size) { hypotheses[it].last() }
    }

    private fun initSortMask() {
        sortMask = IntArray(batchSize) { it }
    }

    private fun updateState(samples: IntArray, sampleScores: DoubleArray, stepLogProbs: DoubleArray, sortMask: IntArray) {
        sortState(sortMask)

        scores = sampleScores.toMutableList()
        for (i in hypotheses.indices) {
            hypotheses[i].add(samples[i])  // hypotheses is 5 of one list, should be a copy
            eachStepProbs[i].add(stepLogProbs[i])  // same
        }
        stashTerminated(samples)
    }

    private fun stashTerminated(samples: IntArray) {
        val toStash = isSampleTerminates(samples.copyOfRange(0, searchSize))

        val normScores = getNormalizedScores().apply { for (i in this.indices) this[i] = kotlin.math.exp(this[i]) }

        val trimmedHypotheses = hypotheses.subList(0, searchSize)
        val trimmedScores = normScores.copyOfRange(0, searchSize)

        for (i in trimmedHypotheses.indices.filter { toStash[it] }) {
            assert(trimmedHypotheses[i].size == length.toInt())
            terminatedHypotheses.add(Pair(ArrayList(trimmedHypotheses[i]), trimmedScores[i]))
        }

        val terminated = isSampleTerminates(samples)
        val notTerminatedInds = (terminated.indices).filter { !terminated[it] }.toIntArray()
        applySliceToState(notTerminatedInds)
        sortState()
}

    private fun sortState(sortMask: IntArray? = null) {
        if (sortMask == null) {
            applySliceToState(topk1d(scores.toDoubleArray(), min(searchSize, scores.size)))
        } else {
            applySliceToState(sortMask)
        }
    }

    private fun isSampleTerminates(samples: IntArray): BooleanArray {
        return BooleanArray(samples.size) { samples[it] in eosIdsSet }
    }

    private fun applySliceToState(tensorSlice: IntArray) {
        scores = scores.slice(tensorSlice).toMutableList()
//        hypotheses = hypotheses.slice(tensorSlice)
        hypotheses = tensorSlice.map { ArrayList(hypotheses[it]) }
//        eachStepProbs = eachStepProbs.slice(tensorSlice)
        eachStepProbs = tensorSlice.map { ArrayList(eachStepProbs[it]) }
        if (sortMask != null) {
            sortMask = sortMask!!.sliceArray(tensorSlice)
        }
    }

    private fun getNormalizedScores(): DoubleArray {
        val normFactor = ((lenNormBase + length) / (lenNormBase + 1)).pow(lenNormPow)
        return DoubleArray(scores.size) { scores[it] / normFactor }
    }
}
