package org.jetbrains.research.kotlin.mpp.inference.graph

import GraphProto
import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.types.ValueInfo

//TODO: support general graphs
//TODO: check i/o tensor shapes explicitly
//TODO: graph optimizations (i.e. remove "Identity" nodes, fuse "MatMul" with "Add" etc)
class Graph(
    initializers: List<Tensor<*>>,
    val nodes: List<Node>,
    val input: List<ValueInfo>,
    val output: List<ValueInfo>,
    val info: List<ValueInfo>
) {
    init {
        val initNodes = NodeIO().addNotNullValues(initializers)

        for (node in nodes) {
            val constInitializers = initNodes.filterKeys { it in node.inputs.keys }
            node.inputs.putAll(constInitializers)
            node.setMutable(node.inputs.keys - constInitializers.keys)
        }
    }

    val availableInputs: List<String>
        get() = nodes.flatMap { it.inputs.availableInputs.keys }

    inline fun <reified T : Number> setInput(name: String, value: List<T>) {
        require(name in availableInputs)
        nodes.findNode(name).inputs[name] = Tensor(value, TensorProto.DataType.FLOAT)
    }

    inline fun <reified T : Number> setInput(value: List<T>) {
        require(input.size == 1) { "Specify input node name explicitly" }
        val name = input.single().name
        setInput(name, value)
    }

    //only for sequential models
    fun run() {
        //TODO: check that all inputs were set and not null
        nodes.zipWithNext { current, next ->
            current.process()
            next.inputs.putAll(current.outputs)
        }
        nodes.last().process()
    }

    private fun fetchOutputs() {
        nodes.last().outputs
    }

    companion object {
        fun build(proto: GraphProto): Graph {
            val initializers = proto.initializer.map { Tensor.create(it) }

            val inputs = proto.input.map { ValueInfo.create(it) }
            val outputs = proto.output.map { ValueInfo.create(it) }

            val nodes = proto.node.map { Node(it) }

            val info = proto.value_info.map { ValueInfo.create(it) }
            return Graph(initializers, nodes, inputs, outputs, info)
        }

        fun List<Node>.findNode(input: String): Node {
            return this.find { input in it.inputs.keys }!!
        }
    }
}
