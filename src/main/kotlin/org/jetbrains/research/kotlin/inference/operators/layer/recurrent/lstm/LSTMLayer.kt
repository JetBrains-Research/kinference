package org.jetbrains.research.kotlin.inference.operators.layer.recurrent.lstm

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.math.*
import org.jetbrains.research.kotlin.inference.math.extensions.allocateNDArray
import org.jetbrains.research.kotlin.inference.math.extensions.splitWithAxis
import org.jetbrains.research.kotlin.inference.onnx.TensorProto.DataType
import org.jetbrains.research.kotlin.inference.operators.activations.Activation

open class LSTMLayer(hiddenSize: Int, activations: List<String>, direction: String) : LSTMBase(hiddenSize, activations, direction) {

    private var lstmData: LSTMData? = null

    init {
        require(direction == "forward" || direction == "reverse")
        require(activations.size >= 3)
    }

    override fun apply(inputs: List<NDArray>, sequenceLens: IntArray, outputArray: MutableNDArray, startOffset: Int): List<Tensor> {
        val batchSize = batchSize!!
        val seqLength = seqLength!!
        val type = type!!
        val lstmData = lstmData!!

        val (f, g, h) = activations.map { Activation.create(it, type) }

        var currentOffset = if (direction == "forward") startOffset else outputArray.linearSize - startOffset
        val stepOffset = outputArray.strides.strides[0]

        val gatesData = GatesData.allocateGates(hiddenSize, type)
        val lastStates = State.create(lstmData.initialOutput, lstmData.initialCellState, batchSize, hiddenSize, type)

        var batchNum = if (direction == "forward") 0 else seqLength - 1
        for (i in 0 until seqLength) {
            val temp = batchNum * batchSize
            for (inputNum in 0 until batchSize) {
                if (batchNum >= sequenceLens[inputNum]) continue
                step(lstmData, inputs[temp + inputNum], outputArray, currentOffset + hiddenSize * inputNum, gatesData, lastStates[inputNum], f, g, h)
            }
            if (direction == "forward") {
                currentOffset += stepOffset
                batchNum++
            } else {
                currentOffset -= stepOffset
                batchNum--
            }
        }
        val lastState = lastStates.toOutput()

        return listOf(outputArray.asTensor(), lastState.output.asTensor(), lastState.cellState.asTensor())
    }

    private fun NDArray.activateStep(lastState: State, lstmWeight: MutableNDArray, lstmGate: MutableNDArray, activation: PrimitiveToPrimitiveFunction,
                                     recurrent: NDArray, bias: NDArray?, peepholes: NDArray? = null) {
        this as NumberNDArray; lstmWeight as MutableNumberNDArray; lstmGate as MutableNumberNDArray
        lstmGate.clean()
        this.dot(lstmWeight, lstmGate)
        if (!lastState.isOutputZero) (lastState.output as NumberNDArray).dot(recurrent as NumberNDArray, lstmGate)
        if (!lastState.isCellStateZero && peepholes != null)
            lstmGate.plusAssign(peepholes as NumberNDArray * lastState.cellState as NumberNDArray)
        if (bias != null) lstmGate.plusAssign(bias)
        lstmGate.mapMutable(activation)
    }

    private fun step(lstmData: LSTMData, input: NDArray, output: MutableNDArray, outputOffset: Int, gatesData: GatesData,
                     lastState: State, f: PrimitiveToPrimitiveFunction, g: PrimitiveToPrimitiveFunction, h: PrimitiveToPrimitiveFunction) {
        input.activateStep(lastState, lstmData.weights.input, gatesData.input, f, lstmData.recurrentWeights.input, lstmData.bias?.input, lstmData.peepholes?.input)
        input.activateStep(lastState, lstmData.weights.forget, gatesData.forget, f, lstmData.recurrentWeights.forget, lstmData.bias?.forget, lstmData.peepholes?.forget)
        input.activateStep(lastState, lstmData.weights.cellGate, gatesData.cellGate, g, lstmData.recurrentWeights.cellGate, lstmData.bias?.cellGate)

        if (!lastState.isCellStateZero) (lastState.cellState as MutableNumberNDArray).timesAssign(gatesData.forget)
        (gatesData.input as MutableNumberNDArray).timesAssign(gatesData.cellGate)
        (lastState.cellState as MutableNumberNDArray).plusAssign(gatesData.input)

        input.activateStep(lastState, lstmData.weights.output, gatesData.output, f, lstmData.recurrentWeights.output, lstmData.bias?.output, lstmData.peepholes?.output)

        lastState.output = lastState.cellState.map(h).apply { timesAssign(gatesData.output) }

        output.placeAllFrom(outputOffset, lastState.output)

        lastState.isOutputZero = false
        lastState.isCellStateZero = false
    }

    override fun parseTempInputs(weights: Tensor, recurrentWeights: Tensor, bias: Tensor?, initialOutput: Tensor?, initialCellState: Tensor?, peepholes: Tensor?) {
        if (lstmData == null) {
            val parsedWeights = GatesData.createWeights(weights.data.toMutable())
            val parsedRecurrentWeights = GatesData.createWeights(recurrentWeights.data.toMutable())
            lstmData = LSTMData(parsedWeights, parsedRecurrentWeights, null, null, null, null, type!!)

            this.weights = weights.data
            this.recurrentWeights = recurrentWeights.data
            this.bias = null
            this.initialOutput = null
            this.initialCellState = null
            this.peepholes = null
        }
        if (weights.data !== this.weights) {
            lstmData = lstmData!!.updateWeights(GatesData.createWeights(weights.data.toMutable()))
            this.weights = weights.data
        }
        if (recurrentWeights.data !== this.recurrentWeights) {
            lstmData = lstmData!!.updateRecurrentWeights(GatesData.createWeights(recurrentWeights.data.toMutable()))
            this.recurrentWeights = recurrentWeights.data
        }
        if (bias != null && bias.data !== this.bias) {
            lstmData = lstmData!!.updateBias(GatesData.createBias(bias.data.toMutable()))
            this.bias = bias.data
        }
        if (initialOutput != null && initialOutput.data !== this.initialOutput) {
            lstmData = lstmData!!.updateInitialOutput(initialOutput.data.toMutable().squeeze(0).splitWithAxis(batchSize!!))
            this.initialOutput = initialOutput.data
        }
        if (initialCellState != null && initialCellState.data !== this.initialCellState) {
            lstmData = lstmData!!.updateInitialCellGate(initialCellState.data.toMutable().squeeze(0).splitWithAxis(batchSize!!))
            this.initialCellState = initialCellState.data
        }
        if (peepholes != null && peepholes.data !== this.peepholes) {
            lstmData = lstmData!!.updatePeepholes(GatesData.createPeepholes(peepholes.data.toMutable()))
            this.peepholes = peepholes.data
        }
    }

    protected fun Array<State>.toOutput(): State {
        val strides = Strides(intArrayOf(1, batchSize!!, hiddenSize))
        val outputArray = allocateNDArray(type!!, strides)
        val cellStateArray = allocateNDArray(type!!, strides)

        for (i in this.indices) {
            val offset = i * hiddenSize
            outputArray.placeAllFrom(offset, this[i].output)
            cellStateArray.placeAllFrom(offset, this[i].cellState)
        }

        return State(outputArray, cellStateArray, false, false)
    }

    companion object {
        fun create(hiddenSize: Int, activations: List<String>, direction: String, lstmData: LSTMData, seqLength: Int, batchSize: Int, type: DataType): LSTMLayer {
            val lstm = LSTMLayer(hiddenSize, activations, direction)
            lstm.lstmData = lstmData
            lstm.seqLength = seqLength
            lstm.batchSize = batchSize
            lstm.type = type.resolveLocalDataType()
            return lstm
        }
    }
}


