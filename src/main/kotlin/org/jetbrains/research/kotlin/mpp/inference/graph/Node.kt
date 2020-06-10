package org.jetbrains.research.kotlin.mpp.inference.graph

import NodeProto
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.OperatorFactory

class Node(proto: NodeProto, graph: Graph) {
    private val inputs = proto.input
    private val outputs = proto.output
    private val operator = OperatorFactory.create(proto.op_type, proto.attribute.map { Attribute.create(it, graph) }.associateBy(Attribute<Any>::name))

    fun execute(context: Context) {
        operator.applyWithCheck(inputs.map { input -> context.getValue(input) }).zip(outputs) { tensor, output ->
            if (output.isNotBlank()) {
                context.putValue(output, tensor.copy(name = output))
            }
        }
    }
}
