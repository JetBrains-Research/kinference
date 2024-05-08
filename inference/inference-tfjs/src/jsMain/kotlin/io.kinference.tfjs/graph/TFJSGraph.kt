package io.kinference.tfjs.graph

import io.kinference.graph.*
import io.kinference.operator.Operator
import io.kinference.operator.OperatorSetRegistry
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.operators.TFJSOperatorFactory
import io.kinference.utils.closeAll

class TFJSGraph private constructor(
    proto: GraphProto,
    initializers: ArrayList<TFJSData<*>>,
    operators: ArrayList<Operator<TFJSData<*>, TFJSData<*>>>,
    valueOrderInfo: GraphValueOrderInfo,
    private val preparedTensorsContext: GraphContext<TFJSData<*>> = GraphContext()
) : Graph<TFJSData<*>>(proto, initializers, operators, valueOrderInfo) {

    override suspend fun close() {
        preparedTensorsContext.close()
        super.close()
    }

    fun addTensorToContext(tensor: TFJSTensor) {
        preparedTensorsContext.putValue(tensor.name!!, tensor)
    }

    override fun makeContext(root: GraphContext<TFJSData<*>>?): GraphContext<TFJSData<*>> {
        val context = GraphContext(root)
        context.mergeContext(preparedTensorsContext)
        return context
    }

    companion object {
        suspend operator fun invoke(proto: GraphProto, opSetRegistry: OperatorSetRegistry): TFJSGraph {
            val valueOrderInfo = GraphValueOrderInfo()
            val nodes = proto.collectOperators<TFJSData<*>>(valueOrderInfo)
            val operators = ArrayList<Operator<TFJSData<*>, TFJSData<*>>>(nodes.size).apply {
                try {
                    for (node in nodes) {
                        add(TFJSOperatorFactory.create(node.proto, opSetRegistry))
                    }
                } catch (e: Exception) {
                    closeAll(this)
                    throw e
                }
            }

            val initializers = getInitializers(proto) { tensorProto ->
                TFJSTensor.create(tensorProto) as TFJSData<*>
            }
            return TFJSGraph(proto, initializers, operators, valueOrderInfo)
        }
    }
}
