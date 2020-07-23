package org.jetbrains.research.kotlin.inference.operators

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ONNXData
import org.jetbrains.research.kotlin.inference.operators.activations.*
import org.jetbrains.research.kotlin.inference.operators.flow.Loop
import org.jetbrains.research.kotlin.inference.operators.layer.recurrent.lstm.LSTM
import org.jetbrains.research.kotlin.inference.operators.math.Add
import org.jetbrains.research.kotlin.inference.operators.math.MatMul
import org.jetbrains.research.kotlin.inference.operators.seq.ConcatFromSequence
import org.jetbrains.research.kotlin.inference.operators.seq.SplitToSequence
import org.jetbrains.research.kotlin.inference.operators.tensor.*

object OperatorFactory {
    @Suppress("UNCHECKED_CAST")
    fun create(name: String?, attributes: Map<String, Attribute<Any>>, usedOutputsNum: Int) = when (name) {
        "Add" -> Add(attributes, usedOutputsNum)
        "Cast" -> Cast(attributes, usedOutputsNum)
        "Concat" -> Concat(attributes, usedOutputsNum)
        "ConcatFromSequence" -> ConcatFromSequence(attributes, usedOutputsNum)
        "Constant" -> Constant(attributes, usedOutputsNum)
        "Gather" -> Gather(attributes, usedOutputsNum)
        "Identity" -> Identity(attributes, usedOutputsNum)
        "LSTM" -> LSTM(attributes, usedOutputsNum)
        "Loop" -> Loop(attributes, usedOutputsNum)
        "MatMul" -> MatMul(attributes, usedOutputsNum)
        "Relu" -> Relu(attributes, usedOutputsNum)
        "Reshape" -> Reshape(attributes, usedOutputsNum)
        "Sigmoid" -> Sigmoid(attributes, usedOutputsNum)
        "Softmax" -> Softmax(attributes, usedOutputsNum)
        "Split" -> Split(attributes, usedOutputsNum)
        "SplitToSequence" -> SplitToSequence(attributes, usedOutputsNum)
        "Squeeze" -> Squeeze(attributes, usedOutputsNum)
        "Tanh" -> Tanh(attributes, usedOutputsNum)
        "Transpose" -> Transpose(attributes, usedOutputsNum)
        else -> error("Unsupported operator $name")
    } as Operator<ONNXData, ONNXData>
}
