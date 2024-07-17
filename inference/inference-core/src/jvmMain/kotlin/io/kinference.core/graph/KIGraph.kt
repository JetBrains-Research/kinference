package io.kinference.core.graph

import io.kinference.core.*
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.operators.KIOperatorFactory
import io.kinference.graph.*
import io.kinference.operator.Operator
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.GraphProto


class KIGraph private constructor(
    proto: GraphProto,
    initializers: ArrayList<KIONNXData<*>>,
    operators: ArrayList<Operator<KIONNXData<*>, KIONNXData<*>>>,
    valueOrderInfo: GraphValueOrderInfo,
    private val preparedTensorsContext: GraphContext<KIONNXData<*>> = GraphContext()
) : Graph<KIONNXData<*>>(proto, initializers, operators, valueOrderInfo) {
    override fun makeContext(root: GraphContext<KIONNXData<*>>?): GraphContext<KIONNXData<*>> {
        val context = GraphContext(root)
        context.mergeContext(preparedTensorsContext)
        return context
    }

    fun addTensorToContext(tensor: KITensor) {
        preparedTensorsContext.putValue(tensor.name!!, tensor)
    }

    companion object {
        suspend operator fun invoke(proto: GraphProto, opSetRegistry: OperatorSetRegistry): KIGraph {
            val valueOrderInfo = GraphValueOrderInfo()
            val nodes = proto.collectOperators<KIONNXData<*>>(valueOrderInfo)
            val operators = ArrayList<Operator<KIONNXData<*>, KIONNXData<*>>>(nodes.size).apply {
                for (node in nodes) {
                    add(KIOperatorFactory.create(node.proto, opSetRegistry))
                }
            }

            val initializers = getInitializers(proto) { tensorProto ->
                KITensor.create(tensorProto) as KIONNXData<*>
            }
            return KIGraph(proto, initializers, operators, valueOrderInfo)
        }
    }
}
