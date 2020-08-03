package org.jetbrains.research.kotlin.inference.graph

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.onnx.NodeProto
import org.jetbrains.research.kotlin.inference.operators.OperatorFactory

class Node(proto: NodeProto, graph: Graph) {
    private val inputs = proto.input
    private val outputs = proto.output
    private val operator = OperatorFactory.create(proto.op_type, proto.attribute.map { Attribute.create(it, graph) }.associateBy(Attribute<Any>::name), outputs.size)

    fun execute(context: Context) {
        operator.applyWithCheck(inputs.map { input -> if (input.isEmpty()) null else context.getValue(input) }).zip(outputs) { data, output ->
            if (data == null) require(output.isEmpty()) { "Required output '$output' not provided by '${operator.info.name}' operator" }
            if (output.isNotEmpty()) {
                context.putValue(output, data!!.clone(newName = output))
            }
        }
    }
}
