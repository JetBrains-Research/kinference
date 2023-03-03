package io.kinference.tfjs.graph

import io.kinference.graph.*
import io.kinference.operator.Operator
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.operators.TFJSOperatorFactory

class TFJSGraph(
    proto: GraphProto, operators: ArrayList<Operator<TFJSData<*>, TFJSData<*>>>,
    valueOrderInfo: GraphValueOrderInfo,
) : Graph<TFJSData<*>>(proto, operators, valueOrderInfo) {
    override fun prepareInput(proto: TensorProto): TFJSData<*> = TFJSTensor.create(proto)
    override fun makeContext(root: GraphContext<TFJSData<*>>?): GraphContext<TFJSData<*>> = GraphContext(root)

    companion object {
        suspend operator fun invoke(proto: GraphProto, opSetRegistry: OperatorSetRegistry): TFJSGraph {
            val valueOrderInfo = GraphValueOrderInfo()
            val nodes = proto.collectOperators<TFJSData<*>>(valueOrderInfo)
            val operators = ArrayList<Operator<TFJSData<*>, TFJSData<*>>>(nodes.size).apply {
                for (node in nodes) {
                    add(TFJSOperatorFactory.create(node.proto, opSetRegistry))
                }
            }

            return TFJSGraph(proto, operators, valueOrderInfo)
        }
    }
}
