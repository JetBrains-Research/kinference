package org.jetbrains.research.kotlin.mpp.inference.graph

import GraphProto
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
        val initNodes = NodeIO().addTensors(initializers)

        for (node in nodes) {
            val constInitializers = initNodes.filterNames { it in node.inputs.names }
            node.inputs.addNamedTensors(constInitializers)
            node.setMutable(node.inputs.names - constInitializers.keys)
        }
    }

    val availableInputs: List<String>
        get() = nodes.filter { it.type == Node.NodeType.INPUT }.flatMap { it.inputs.availableForWriting }

    inline fun <reified T : Number> setInput(name: String, value: List<T>): Graph {
        require(name in availableInputs) { "Required input node is either already set or not found" }

        val type = input.find { it.name == name }?.type
        requireNotNull(type)

        nodes.find { name in it.inputs.names }!!.inputs[name] = Tensor(value, type)
        return this
    }

    inline fun <reified T : Number> setInput(value: List<T>): Graph {
        require(input.size == 1) { "Multiple input nodes found. Specify input name explicitly" }
        val name = input.single().name
        return setInput(name, value)
    }

    fun fetchOutputs(): List<Tensor<*>?> {
        val out = nodes.filter { it.type == Node.NodeType.OUTPUT }
        return out.flatMap { it.outputs.tensors.toList() }
    }

    //only for sequential models
    fun execute(): Graph {
        //TODO: check that all inputs were set and not null
        nodes.zipWithNext { current, next ->
            current.execute()
            next.inputs.merge(current.outputs)
        }
        nodes.last().execute()
        return this
    }

    companion object {
        fun build(proto: GraphProto): Graph {
            val initializers = proto.initializer.map { Tensor.create(it) }

            val inputs = proto.input.map { ValueInfo.create(it) }
            val outputs = proto.output.map { ValueInfo.create(it) }

            val nodes = proto.node.map {
                val type = when {
                    it.input.any { name -> inputs.containsName(name) } -> Node.NodeType.INPUT
                    it.output.any { name -> outputs.containsName(name) } -> Node.NodeType.OUTPUT
                    else -> Node.NodeType.INNER
                }
                Node(it, type)
            }

            val info = proto.value_info.map { ValueInfo.create(it) }
            return Graph(initializers, nodes, inputs, outputs, info)
        }

        private fun List<ValueInfo>.containsName(name: String): Boolean {
            return this.map { it.name }.contains(name)
        }
    }
}
