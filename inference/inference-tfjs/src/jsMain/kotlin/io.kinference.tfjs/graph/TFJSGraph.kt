package io.kinference.tfjs.graph

import io.kinference.graph.*
import io.kinference.operator.Operator
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.operators.TFJSOperatorFactory
import io.kinference.tfjs.operators.layer.recurrent.gru.GRUContext
import io.kinference.tfjs.operators.layer.recurrent.lstm.LSTMContext
import io.kinference.utils.closeAll

class TFJSGraph(
    proto: GraphProto,
    operators: ArrayList<Operator<TFJSData<*>, TFJSData<*>>>,
    valueOrderInfo: GraphValueOrderInfo,
    private val preparedTensorsContext: GraphContext<TFJSData<*>> = GraphContext()
) : Graph<TFJSData<*>>(proto, operators, valueOrderInfo) {

    override fun close() {
        preparedTensorsContext.close()
        super.close()
    }

    override fun prepareInput(proto: TensorProto): TFJSData<*> = TFJSTensor.create(proto)
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

            val graph = TFJSGraph(proto, operators, valueOrderInfo)
            for (operator in graph.operators) {
                when(operator.info.type) {
                    "LSTM" -> LSTMContext.appendContext(graph.preparedTensorsContext, graph.initializers as List<TFJSTensor>, operator)
                    "GRU" -> GRUContext.appendContext(graph.preparedTensorsContext, graph.initializers as List<TFJSTensor>, operator)
                }
            }
            return graph
        }
    }
}
