package io.kinference.tfjs.operators.layer.recurrent.gru

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.ndarray.unstackAs3DTypedArray
import io.kinference.ndarray.update
import io.kinference.tfjs.operators.activations.activate
import io.kinference.utils.Closeable
import io.kinference.utils.closeArrays

class GRUDefaultGate internal constructor(
    private val weights: NumberNDArrayTFJS,
    private val recurrentWeights: NumberNDArrayTFJS,
    private val bias: NumberNDArrayTFJS?,
    batchSize: Int,
    hiddenSize: Int,
) : Closeable {
    private val gateData: Array<MutableNumberNDArrayTFJS> = Array(batchSize) { NDArrayTFJS.floatZeros(arrayOf(hiddenSize)).asMutable() }

    suspend fun compute(
        input: NumberNDArrayTFJS,
        hiddenState: GRUHiddenState,
        activationFunction: String,
        numDirection: Int,
        batchNum: Int
    ) {
        val newGateData = input.dot(weights)
        newGateData.plusAssign(hiddenState.getVector(numDirection, batchNum).dot(recurrentWeights))
        if (bias != null) newGateData.plusAssign(bias)

        gateData.update(batchNum, newGateData.activate(activationFunction).asMutable())
    }

    fun getVector(batchNum: Int) = gateData[batchNum]

    override suspend fun close() {
        closeArrays(gateData)
    }
}

class GRUHiddenGate private constructor(
    private val weights: NumberNDArrayTFJS,
    private val recurrentWeights: NumberNDArrayTFJS,
    private val bias: NumberNDArray?,
    private val weightsBias: NumberNDArray?,
    private val recurrentBias: NumberNDArray?,
    private val linearBeforeReset: Boolean,
    private val gateData: Array<MutableNumberNDArrayTFJS>,
) : Closeable {

    override suspend fun close() {
        closeArrays(gateData)
    }

    companion object {
        internal suspend operator fun invoke(
            weights: NumberNDArrayTFJS,
            recurrentWeights: NumberNDArrayTFJS,
            wBias: NumberNDArrayTFJS?,
            rBias: NumberNDArrayTFJS?,
            batchSize: Int,
            hiddenSize: Int,
            linearBeforeReset: Boolean
        ): GRUHiddenGate {
            val bias: NumberNDArrayTFJS?
            val weightsBias: NumberNDArrayTFJS?
            val recurrentBias: NumberNDArrayTFJS?

            if (linearBeforeReset) {
                bias = null
                weightsBias = wBias
                recurrentBias = rBias
            } else {
                bias = when {
                    wBias != null && rBias != null -> wBias.plus(rBias)
                    wBias != null -> wBias
                    rBias != null -> rBias
                    else -> null
                }
                weightsBias = null
                recurrentBias = null
            }

            val gateData = Array(batchSize) { NDArrayTFJS.floatZeros(arrayOf(hiddenSize)).asMutable() }

            return GRUHiddenGate(
                weights, recurrentWeights, bias, weightsBias, recurrentBias, linearBeforeReset, gateData
            )
        }
    }

    suspend fun compute(
        input: NumberNDArrayTFJS,
        hiddenState: GRUHiddenState,
        gates: GRUGates,
        activationFunction: String,
        numDirection: Int,
        batchNum: Int
    ) =
        if (linearBeforeReset)
            computeWithReset(input, hiddenState, gates, activationFunction, numDirection, batchNum)
        else
            computeDefault(input, hiddenState, gates, activationFunction, numDirection, batchNum)

    private suspend fun computeDefault(
        input: NumberNDArrayTFJS,
        hiddenState: GRUHiddenState,
        gates: GRUGates,
        activationFunction: String,
        numDirection: Int,
        batchNum: Int
    ) {
        val newGateData = input.dot(weights)
        val tempData = gates.reset.getVector(batchNum)
            .times(hiddenState.getVector(numDirection, batchNum))
        newGateData.plusAssign(tempData.dot(recurrentWeights))

        if (bias != null) newGateData.plusAssign(bias)

        gateData.update(batchNum, newGateData.activate(activationFunction).asMutable())
        tempData.close()
    }

    private suspend fun computeWithReset(
        input: NumberNDArrayTFJS,
        hiddenState: GRUHiddenState,
        gates: GRUGates,
        activationFunction: String,
        numDirection: Int,
        batchNum: Int
    ) {
        val newGateData = hiddenState.getVector(numDirection, batchNum).dot(recurrentWeights)
        if (recurrentBias != null) newGateData.plusAssign(recurrentBias)
        newGateData.timesAssign(gates.reset.getVector(batchNum))
        newGateData.plusAssign(input.dot(weights))

        if (weightsBias != null) newGateData.plusAssign(weightsBias)

        gateData.update(batchNum, newGateData.activate(activationFunction).asMutable())
    }

    fun getVector(batchNum: Int) = gateData[batchNum]
}

data class GRUGates(val update: GRUDefaultGate, val reset: GRUDefaultGate, val hidden: GRUHiddenGate) : Closeable {

    override suspend fun close() {
        update.close()
        reset.close()
        hidden.close()
    }

    companion object {
        suspend fun create(
            weights: NumberNDArrayTFJS, recurrentWeights: NumberNDArrayTFJS, bias: NumberNDArrayTFJS?,
            batchSize: Int, hiddenSize: Int, linearBeforeReset: Boolean
        ): GRUGates {
            val (weightsUpdate, weightsReset, weightsHidden) = weights.unstack()
            val (recWeightsUpdate, recWeightsReset, recWeightsHidden) = recurrentWeights.unstack()
            val biases = bias?.unstack() ?: arrayOfNulls<NumberNDArrayTFJS?>(6)

            val updateGate = GRUDefaultGate(
                weightsUpdate,
                recWeightsUpdate,
                biases[0]?.plus(biases[3]!!),
                batchSize, hiddenSize
            )

            val resetGate = GRUDefaultGate(
                weightsReset,
                recWeightsReset,
                biases[1]?.plus(biases[4]!!),
                batchSize, hiddenSize
            )

            val hiddenGate = GRUHiddenGate(
                weightsHidden,
                recWeightsHidden,
                biases[2],
                biases[5],
                batchSize, hiddenSize, linearBeforeReset
            )

            return GRUGates(updateGate, resetGate, hiddenGate)
        }
    }

}

class GRUHiddenState internal constructor(
    initHiddenState: NumberNDArrayTFJS?,
    numDirection: Int, batchSize: Int, hiddenSize: Int
) : Closeable {
    private val stateData = if (initHiddenState == null) {
        Array(numDirection) { Array(batchSize) { NDArrayTFJS.floatZeros(arrayOf(hiddenSize)).asMutable() } }
    } else {
        initHiddenState.unstackAs3DTypedArray()
    }

    val data: Array<Array<MutableNumberNDArrayTFJS>>
        get() = stateData

    override suspend fun close() {
        stateData.forEach { closeArrays(it) }
    }

    private suspend fun MutableNumberNDArrayTFJS.reset(): MutableNumberNDArrayTFJS {
        val one = NDArrayTFJS.floatScalar(1.0f)
        return this.unaryMinus().plus(one).also { one.close() }
    }

    suspend fun compute(gates: GRUGates, numDirection: Int, batchNum: Int) {
        val currentStateData = stateData[numDirection][batchNum]
        val tempData = gates.update.getVector(batchNum).reset()

        currentStateData.timesAssign(gates.update.getVector(batchNum))
        tempData.timesAssign(gates.hidden.getVector(batchNum))
        currentStateData.plusAssign(tempData)
        tempData.close()
    }

    fun getVector(numDirection: Int, batchNum: Int) = stateData[numDirection][batchNum]
}

data class GRULayerOutput(
    val output: NumberNDArrayTFJS,
    val hiddenState: NumberNDArrayTFJS
)
