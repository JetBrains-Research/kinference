package io.kinference.ndarray.generating

import io.kinference.ndarray.*
import io.kinference.primitives.types.DataType
import java.lang.Math.*
import kotlin.math.min
import kotlin.math.pow

abstract class Search(val eosIds: IntArray, val vocabSize: Int, val searchSize: Int,
                      val lenNormBase: Int = 0, val lenNormPow: Int = 0, val repetitionPenalty: Double = 1.0) {

    @ExperimentalUnsignedTypes
    abstract fun step(ndStepLogProbs: MutableNumberNDArray, context: List<Int>): List<Int>

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
    abstract fun lastPredictions(): List<Int>

    /**
     * Current batch size
     */
    abstract fun batchSize(): Int
}

class BeamSearch(eosIds: IntArray, vocabSize: Int, searchSize: Int,
                 lenNormBase: Int = 0, lenNormPow: Int = 0, repetitionPenalty: Double = 1.0) :
    Search(eosIds, vocabSize, searchSize, lenNormBase, lenNormPow, repetitionPenalty) {

    private val length = 1.0

    private var scores: MutableList<Double> = ArrayList()
    private var hypotheses: List<MutableList<Int>> = ArrayList()
    private var eachStepProbs: List<MutableList<Double>> = ArrayList()
    private val terminatedHypotheses: MutableList<Pair<List<Int>, Double>> = ArrayList()
    private var sortMask: List<Int>? = null
    private val eosIdsSet: Set<Int> = eosIds.toSet()

    private fun initState(type: DataType) {
//        scores = torch.zeros(1, dtype = type)
//        hypotheses = torch.empty(1, 0, dtype = torch.long)
//        each_step_probs = torch.empty(1, 0, dtype = dtype)
//        eos_tensor = torch.tensor(self._eos_ids, dtype = torch.long).unsqueeze(1)
    }

    private fun toDoubleList2d(data: NumberNDArray): List<MutableList<Double>> {
        assert(data.shape.size == 2)

        val result = ArrayList<MutableList<Double>>()
        for (i in 0 until data.shape[0]) {

            val row = ArrayList<Double>()
            for (j in 0 until data.shape[1]) {
                row.add(data[intArrayOf(i, j)] as Double)
            }

            result.add(row)
        }

        return result
    }

    @ExperimentalUnsignedTypes
    override fun step(ndStepLogProbs: MutableNumberNDArray, context: List<Int>): List<Int> {
        modifyScore(ndStepLogProbs, context)
        super.stepCheck(ndStepLogProbs)

//        if (scores == null) {
//            initState(stepLogProbs.type)
//        }

        val stepLogProbs = toDoubleList2d(ndStepLogProbs)

        val ndScores = DoubleNDArray(scores.toDoubleArray())
        val ndLogProbs = ndStepLogProbs.plus(ndScores)
        val logProbs = toDoubleList2d(ndLogProbs).flatten()

        val flattenStepLogProbs = stepLogProbs.map { it.map { e -> kotlin.math.exp(e) }.toMutableList() }.flatten()

        var samples = topk1dList(logProbs, min((1 + eosIds.size) * searchSize, ndLogProbs.shape[0]))

        val samplesStepLogProbs = samples.map { flattenStepLogProbs[it] }
        val sortMask = samples.map { floorDiv(it, vocabSize) }
        samples = samples.map { floorMod(it, vocabSize) }

        initSortMask()
        val sampleScores: MutableList<Double> = samples.map { logProbs[it] }.toMutableList()
        updateState(samples, sampleScores, samplesStepLogProbs, sortMask)
        length.plus(1)

        return sortMask
    }

    private fun topk1dList(data: List<Double>, size: Int): List<Int> {
        val pairedData = ArrayList<Pair<Double, Int>>()
        for (i in data.indices) {
            pairedData.add(Pair(data[i], i))
        }
        pairedData.sortBy { -it.first }
        return pairedData.map { it.second }.subList(0, size)
    }

    @ExperimentalUnsignedTypes
    private fun modifyScore(scores: MutableNumberNDArray, context: List<Int>) {
        // repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)

        if (repetitionPenalty != 1.0) {
            for (i in 0 until scores.shape[0]) {
                pessimizeScore(scores, i, context.toSet())
            }

            for (i in hypotheses.indices) {
                pessimizeScore(scores, i, hypotheses[i].toSet())
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
        val score = getNormalizedScores().map { kotlin.math.exp(it) }
        for (i in hypotheses.indices) {
            if (mask[i]) {
                ans.add(Pair(hypotheses[i], GenerationInfo(eachStepProbs[i], score[i])))
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
        val score = getNormalizedScores().map { kotlin.math.exp(it) }
        for (i in hypotheses.indices) {
            ans.add(Pair(hypotheses[i], GenerationInfo(eachStepProbs[i], score[i])))
        }

        // ans.sortBy { -it.second.score }
        return listOf(ans)
    }

    override fun lastPredictions(): List<Int> {
        assert(hypotheses.isNotEmpty() && hypotheses[0].size > 0) {"Can't get last predictions if no steps have been performed"}
        return hypotheses.map{ it[it.size - 1] }
    }

    override fun batchSize(): Int {
        return scores.size
    }

    private fun initSortMask() {
        sortMask = (0 until batchSize()).toList()
    }

    private fun updateState(samples: List<Int>, sampleScores: MutableList<Double>, stepLogProbs: List<Double>, sortMask: List<Int>) {
        sortState(sortMask)

        scores = sampleScores
        for (i in hypotheses.indices) {
            hypotheses[i].add(samples[i])
            eachStepProbs[i].add(stepLogProbs[i])
        }
        stashTerminated(samples)
    }

    private fun stashTerminated(samples: List<Int>) {
        val toStash = isSampleTerminates(samples.subList(0, searchSize))

        scores = getNormalizedScores().map { kotlin.math.exp(it) }.toMutableList()

        val trimmedHypotheses = hypotheses.subList(0, searchSize)
        val trimmedScores = scores.subList(0, searchSize)

        for (i in trimmedHypotheses.indices.filter { toStash[it] }) {
            assert(trimmedHypotheses[i].size == length.toInt())
            terminatedHypotheses.add(Pair(ArrayList(trimmedHypotheses[i]), trimmedScores[i]))
        }

        val terminated = isSampleTerminates(samples)
        val notTerminatedInds = (terminated.indices).filter { !terminated[it] }
        applySliceToState(notTerminatedInds)
        sortState()
}

    private fun sortState(sortMask: List<Int>? = null) {
        if (sortMask == null) {
            applySliceToState(topk1dList(scores, min(searchSize, scores.size)))
        } else {
            applySliceToState(sortMask)
        }
    }

    private fun isSampleTerminates(samples: List<Int>): List<Boolean> {
        return samples.map { it in eosIdsSet }
    }

    private fun applySliceToState(tensorSlice: List<Int>) {
        scores = tensorSlice.map { scores[it] }.toMutableList()
        hypotheses = tensorSlice.map { hypotheses[it] }
        eachStepProbs = tensorSlice.map { eachStepProbs[it] }
        if (sortMask != null) {
            sortMask = tensorSlice.map { sortMask!![it] }
        }
    }

    private fun getNormalizedScores(): List<Double> {
        val normFactor = ((lenNormBase + length) / (lenNormBase + 1)).pow(lenNormPow)
        return scores.map { it / normFactor }
    }
}
