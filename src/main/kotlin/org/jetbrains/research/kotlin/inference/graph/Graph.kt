package org.jetbrains.research.kotlin.inference.graph

import org.jetbrains.research.kotlin.inference.data.ONNXData
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.onnx.GraphProto
import org.jetbrains.research.kotlin.inference.types.ValueInfo

//TODO: support general graphs
//TODO: check i/o tensor shapes explicitly
//TODO: graph optimizations (i.e. remove "Identity" nodes, fuse "MatMul" with "Add" etc)
class Graph(proto: GraphProto, parent: Graph? = null) {
    private val rootContext: Context = Context(parent?.context)
    private val context: Context = Context(rootContext)

    val nodes: List<Node>
    val inputs: List<ValueInfo>
    val outputs: List<ValueInfo>
    val info: List<ValueInfo>

    init {
        val initializers = proto.initializer.map { Tensor.create(it) }
        for (tensor in initializers) {
            rootContext.putValue(tensor.info.name, tensor)
        }

        inputs = proto.input.map { ValueInfo.create(it) }
        outputs = proto.output.map { ValueInfo.create(it) }

        nodes = proto.node.map { Node(it, this) }

        info = proto.value_info.map { ValueInfo.create(it) }
    }

    val availableInputs: List<String>
        get() = inputs.map { it.name }

    fun setInput(name: String, value: List<Any>): Graph {
        require(name in availableInputs) { "Required input node not found" }

        val type = inputs.find { it.name == name }?.type
        requireNotNull(type)

        context.putValue(name, Tensor(value, type))
        return this
    }

    fun setInput(value: List<Any>): Graph {
        require(inputs.size == 1) { "Multiple input nodes found. Specify input name explicitly" }
        val name = inputs.single().name
        return setInput(name, value)
    }

    fun setInput(tensor: ONNXData): Graph {
        val name = tensor.info.name

        require(name in availableInputs) { "Required input node not found" }

        context.putValue(name, tensor)
        return this
    }

    //only for sequential models
    fun execute(): List<ONNXData> {
        //TODO: check that all inputs were set and not null

        for (node in nodes) {
            node.execute(context)
        }

        val result = outputs.map { context.getValue(it.name) }
        context.clear()

        return result
    }
}
