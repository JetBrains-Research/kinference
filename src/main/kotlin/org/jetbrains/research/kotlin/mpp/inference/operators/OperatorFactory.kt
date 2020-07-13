package org.jetbrains.research.kotlin.mpp.inference.operators

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.data.ONNXData
import org.jetbrains.research.kotlin.mpp.inference.operators.activations.*
import org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm.LSTM
import org.jetbrains.research.kotlin.mpp.inference.operators.math.Add
import org.jetbrains.research.kotlin.mpp.inference.operators.math.MatMul
import org.jetbrains.research.kotlin.mpp.inference.operators.seq.ConcatFromSequence
import org.jetbrains.research.kotlin.mpp.inference.operators.seq.SplitToSequence
import org.jetbrains.research.kotlin.mpp.inference.operators.tensor.*

object OperatorFactory {
    @Suppress("UNCHECKED_CAST")
    fun create(name: String?, attributes: Map<String, Attribute<Any>>, usedOutputsNum: Int) = when (name) {
        "Add" -> Add(attributes, usedOutputsNum)
        "Concat" -> Concat(attributes, usedOutputsNum)
        "ConcatFromSequence" -> ConcatFromSequence(attributes, usedOutputsNum)
        "Constant" -> Constant(attributes, usedOutputsNum)
        "Identity" -> Identity(attributes, usedOutputsNum)
        "LSTM" -> LSTM(attributes, usedOutputsNum)
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
