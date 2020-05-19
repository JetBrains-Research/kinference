package org.jetbrains.research.kotlin.mpp.inference.graph

import NodeProto
import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.Operator
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

class Node(proto: NodeProto, val type: NodeType) {
    val inputs: NodeIO = NodeIO()
    val outputs: NodeIO = NodeIO()
    val attributes = proto.attribute.map { Attribute.create(it) }.associateBy { it.name }
    private val operatorName = proto.op_type!!

    init {
        proto.input.forEach { inputs.addName(it) }
        proto.output.forEach { outputs.addName(it) }
    }

    enum class NodeType {
        INNER,
        INPUT,
        OUTPUT
    }

    private val mutableInputs: HashSet<String> = HashSet()

    fun setMutable(inputMarks: Set<String>) {
        mutableInputs += inputMarks
    }

    private fun clearMutableInputs() {
        mutableInputs.forEach { inputs[it] = null }
    }

    fun execute(): NodeIO {
        outputs.clearValues()

        val out = Operator(operatorName, inputs.tensors.resolveType(), inputs.tensors.requireNoNulls().toList(), attributes).toList()
        for ((index, name) in outputs.names.withIndex()){
            val tensor = out.getOrNull(index)
            if (tensor != null) tensor.name = name
            outputs[name] = tensor
        }

        clearMutableInputs()
        return outputs
    }

    companion object {
        private fun Collection<Tensor<*>?>.resolveType(): TensorProto.DataType? {
            val typesCount = this.mapNotNull { it?.type }.toHashSet().size
            require(typesCount <= 1) { "Tensors of more than one type found" }
            return this.first()?.type
        }
    }
}
