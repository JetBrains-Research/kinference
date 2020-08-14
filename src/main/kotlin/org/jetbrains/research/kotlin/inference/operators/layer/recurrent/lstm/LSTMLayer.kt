package org.jetbrains.research.kotlin.inference.operators.layer.recurrent.lstm

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*
import org.jetbrains.research.kotlin.inference.extensions.primitives.matrixDotInto
import org.jetbrains.research.kotlin.inference.onnx.TensorProto.DataType
import org.jetbrains.research.kotlin.inference.operators.activations.Activation

open class LSTMLayer(hiddenSize: Int, activations: List<String>, direction: String) : LSTMBase(hiddenSize, activations, direction) {

    private var lstmData: LSTMData? = null

    init {
        require(direction == "forward" || direction == "reverse")
        require(activations.size >= 3)
    }

    override fun apply(inputs: List<TypedNDArray<Any>>, sequenceLens: IntArray, outputArray: MutableTypedNDArray<Any>, startOffset: Int): List<Tensor> {
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

    private fun step(lstmData: LSTMData, input: TypedNDArray<Any>, output: MutableTypedNDArray<Any>, outputOffset: Int, gatesData: GatesData,
                     lastState: State, f: PrimitiveArrayFunction, g: PrimitiveArrayFunction, h: PrimitiveArrayFunction) {
        input.matrixDotInto(lstmData.weights.input, gatesData.input, true)
        if (!lastState.isOutputZero) lastState.output.matrixDotInto(lstmData.recurrentWeights.input, gatesData.input, false)
        if (!lastState.isCellStateZero && lstmData.peepholes != null) gatesData.input.plusAssign(lstmData.peepholes.input * lastState.cellState)
        if (lstmData.bias != null) gatesData.input.plusAssign(lstmData.bias.input)
        gatesData.input.mapElements(f)

        input.matrixDotInto(lstmData.weights.forget, gatesData.forget, true)
        if (!lastState.isOutputZero) lastState.output.matrixDotInto(lstmData.recurrentWeights.forget, gatesData.forget, false)
        if (!lastState.isCellStateZero && lstmData.peepholes != null) gatesData.forget.plusAssign(lstmData.peepholes.forget * lastState.cellState)
        if (lstmData.bias != null) gatesData.forget.plusAssign(lstmData.bias.forget)
        gatesData.forget.mapElements(f)

        input.matrixDotInto(lstmData.weights.cellGate, gatesData.cellGate, true)
        if (!lastState.isOutputZero) lastState.output.matrixDotInto(lstmData.recurrentWeights.cellGate, gatesData.cellGate, false)
        if (lstmData.bias != null) gatesData.cellGate.plusAssign(lstmData.bias.cellGate)
        gatesData.cellGate.mapElements(g)

        if (!lastState.isCellStateZero) lastState.cellState.timesAssign(gatesData.forget)
        gatesData.input.timesAssign(gatesData.cellGate)
        lastState.cellState.plusAssign(gatesData.input)

        input.matrixDotInto(lstmData.weights.output, gatesData.output, true)
        if (!lastState.isOutputZero) lastState.output.matrixDotInto(lstmData.recurrentWeights.output, gatesData.output, false)
        if (!lastState.isCellStateZero && lstmData.peepholes != null) gatesData.output.plusAssign(lstmData.peepholes.output * lastState.cellState)
        if (lstmData.bias != null) gatesData.output.plusAssign(lstmData.bias.output)
        gatesData.output.mapElements(f)

        lastState.output = (lastState.cellState.clone()).mapElements(h) as MutableTypedNDArray<Any>
        lastState.output.timesAssign(gatesData.output)

        output.placeAll(outputOffset, lastState.output.array)

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
        val outputArray = allocateNDArray<Any>(type!!, strides)
        val cellStateArray = allocateNDArray<Any>(type!!, strides)

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


