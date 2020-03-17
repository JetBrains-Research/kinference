package org.jetbrains.research.kotlin.mpp.inference.graph

import NodeProto
import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.operators.Operator
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

class Node(proto: NodeProto, val type: NodeType) {
    val inputs: NodeIO = NodeIO()
    val outputs: NodeIO = NodeIO()
    private val operatorName = proto.op_type!!

    init {
        proto.input.forEach { inputs.addValue(it) }
        proto.output.forEach { outputs.addValue(it) }
    }

    enum class NodeType {
        INNER,
        GRAPH_INPUT,
        GRAPH_OUTPUT
    }

    private val mutableInputs: HashSet<String> = HashSet()

    fun setMutable(inputMarks: Set<String>) {
        mutableInputs += inputMarks
    }

    private fun clearMutableInputs() {
        mutableInputs.forEach { inputs[it] = null }
    }

    fun process(): NodeIO {
        outputs.clearValues()

        val out = Operator(operatorName, inputs.values.resolveType(), inputs.values.requireNoNulls().toList()).toList()
        outputs.keys.forEachIndexed { i, name -> outputs[name] = out.getOrNull(i) }

        clearMutableInputs()
        return outputs
    }

    companion object {
        private fun Collection<Tensor<*>?>.resolveType(): TensorProto.DataType? {
            val a = this.mapNotNull { it?.type }.toHashSet().size
            require(a <= 1)
            return this.first()?.type
        }
    }
}
