package org.jetbrains.research.kotlin.inference.operators.layer.recurrent.lstm

import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.allocateNDArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.splitWithAxis
import org.jetbrains.research.kotlin.inference.extensions.primitives.matrixDotInto
import org.jetbrains.research.kotlin.inference.onnx.TensorProto.DataType
import org.jetbrains.research.kotlin.inference.operators.activations.Activation

open class LSTMLayer(hiddenSize: Int, activations: List<String>, direction: String) : LSTMBase(hiddenSize, activations, direction) {

    private var lstmData: LSTMData? = null

    init {
        require(direction == "forward" || direction == "reverse")
        require(activations.size >= 3)
    }

    override fun apply(inputs: List<List<NDArray<Any>>>, sequenceLens: IntArray, outputArray: NDArray<Any>, startOffset: Int): List<Tensor> {
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
            for (inputNum in inputs[batchNum].indices) {
                if (batchNum >= sequenceLens[inputNum]) continue
                step(lstmData, inputs[batchNum][inputNum], outputArray, currentOffset + hiddenSize * inputNum, gatesData, lastStates[inputNum], f, g, h)
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

    private fun step(lstmData: LSTMData, input: NDArray<Any>, output: NDArray<Any>, outputOffset: Int, gatesData: GatesData,
                     lastState: State, f: PrimitiveArrayFunction, g: PrimitiveArrayFunction, h: PrimitiveArrayFunction) {
        input.matrixDotInto(lstmData.weights.input, gatesData.input, true)
        if (!lastState.isOutputZero) lastState.output.matrixDotInto(lstmData.recurrentWeights.input, gatesData.input, false)
        if (!lastState.isCellStateZero && lstmData.peepholes != null) gatesData.input.plus(lstmData.peepholes.input * lastState.cellState, false)
        if (lstmData.bias != null) gatesData.input.plus(lstmData.bias.input, false)
        gatesData.input.mapElements(f, false)

        input.matrixDotInto(lstmData.weights.forget, gatesData.forget, true)
        if (!lastState.isOutputZero) lastState.output.matrixDotInto(lstmData.recurrentWeights.forget, gatesData.forget, false)
        if (!lastState.isCellStateZero && lstmData.peepholes != null) gatesData.forget.plus(lstmData.peepholes.forget * lastState.cellState, false)
        if (lstmData.bias != null) gatesData.forget.plus(lstmData.bias.forget, false)
        gatesData.forget.mapElements(f, false)

        input.matrixDotInto(lstmData.weights.cellGate, gatesData.cellGate, true)
        if (!lastState.isOutputZero) lastState.output.matrixDotInto(lstmData.recurrentWeights.cellGate, gatesData.cellGate, false)
        if (lstmData.bias != null) gatesData.cellGate.plus(lstmData.bias.cellGate, false)
        gatesData.cellGate.mapElements(g, false)

        if (!lastState.isCellStateZero) lastState.cellState.times(gatesData.forget, false)
        gatesData.input.times(gatesData.cellGate, false)
        lastState.cellState.plus(gatesData.input, false)

        input.matrixDotInto(lstmData.weights.output, gatesData.output, true)
        if (!lastState.isOutputZero) lastState.output.matrixDotInto(lstmData.recurrentWeights.output, gatesData.output, false)
        if (!lastState.isCellStateZero && lstmData.peepholes != null) gatesData.output.plus(lstmData.peepholes.output * lastState.cellState, false)
        if (lstmData.bias != null) gatesData.output.plus(lstmData.bias.output, false)
        gatesData.output.mapElements(f, false)

        lastState.output = lastState.cellState.mapElements(h, true)
        lastState.output.times(gatesData.output, false)

        output.placeAll(outputOffset, lastState.output.array)

        lastState.isOutputZero = false
        lastState.isCellStateZero = false
    }

    override fun parseTempInputs(weights: Tensor, recurrentWeights: Tensor, bias: Tensor?, initialOutput: Tensor?, initialCellState: Tensor?, peepholes: Tensor?) {
        if (lstmData == null) {
            val parsedWeights = GatesData.createWeights(weights.data)
            val parsedRecurrentWeights = GatesData.createWeights(recurrentWeights.data)
            lstmData = LSTMData(parsedWeights, parsedRecurrentWeights, null, null, null, null, type!!)

            this.weights = weights.data
            this.recurrentWeights = recurrentWeights.data
            this.bias = null
            this.initialOutput = null
            this.initialCellState = null
            this.peepholes = null
        }
        if (weights.data !== this.weights) {
            lstmData = lstmData!!.updateWeights(GatesData.createWeights(weights.data))
            this.weights = weights.data
        }
        if (recurrentWeights.data !== this.recurrentWeights) {
            lstmData = lstmData!!.updateRecurrentWeights(GatesData.createWeights(recurrentWeights.data))
            this.recurrentWeights = recurrentWeights.data
        }
        if (bias != null && bias.data !== this.bias) {
            lstmData = lstmData!!.updateBias(GatesData.createBias(bias.data))
            this.bias = bias.data
        }
        if (initialOutput != null && initialOutput.data !== this.initialOutput) {
            lstmData = lstmData!!.updateInitialOutput(initialOutput.data.squeeze(0).splitWithAxis(batchSize!!))
            this.initialOutput = initialOutput.data
        }
        if (initialCellState != null && initialCellState.data !== this.initialCellState) {
            lstmData = lstmData!!.updateInitialCellGate(initialCellState.data.squeeze(0).splitWithAxis(batchSize!!))
            this.initialCellState = initialCellState.data
        }
        if (peepholes != null && peepholes.data !== this.peepholes) {
            lstmData = lstmData!!.updatePeepholes(GatesData.createPeepholes(peepholes.data))
            this.peepholes = peepholes.data
        }
    }

    protected fun Array<State>.toOutput(): State {
        val strides = Strides(intArrayOf(1, batchSize!!, hiddenSize))
        val outputArray = allocateNDArray(type!!, strides)
        val cellStateArray = allocateNDArray(type!!, strides)

        for (i in this.indices) {
            val offset = i * hiddenSize
            outputArray.placeAll(offset, this[i].output.array)
            cellStateArray.placeAll(offset, this[i].cellState.array)
        }

        return State(outputArray, cellStateArray, false, false)
    }

    companion object {
        fun create(hiddenSize: Int, activations: List<String>, direction: String, lstmData: LSTMData, seqLength: Int, batchSize: Int, type: DataType): LSTMLayer {
            val lstm = LSTMLayer(hiddenSize, activations, direction)
            lstm.lstmData = lstmData
            lstm.seqLength = seqLength
            lstm.batchSize = batchSize
            lstm.type = type
            return lstm
        }
    }
}


