package io.kinference.core.graph

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.operators.KIOperatorFactory
import io.kinference.core.operators.layer.attention.AttentionContext
import io.kinference.core.operators.layer.attention.QAttentionContext
import io.kinference.core.operators.layer.recurrent.gru.GRUContext
import io.kinference.core.operators.layer.recurrent.lstm.LSTMContext
import io.kinference.core.operators.math.MatMulIntegerVer10
import io.kinference.core.operators.quantization.lstm.DynamicQuantizeLSTMContext
import io.kinference.graph.*
import io.kinference.operator.Operator
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto


class KIGraph private constructor(
    proto: GraphProto,
    operators: ArrayList<Operator<KIONNXData<*>, KIONNXData<*>>>,
    valueOrderInfo: GraphValueOrderInfo,
    private val preparedTensorsContext: GraphContext<KIONNXData<*>> = GraphContext()
) : Graph<KIONNXData<*>>(proto, operators, valueOrderInfo) {
    override fun makeContext(root: GraphContext<KIONNXData<*>>?): GraphContext<KIONNXData<*>> {
        val context = GraphContext(root)
        context.mergeContext(preparedTensorsContext)
        return context
    }

    override fun prepareInput(proto: TensorProto): KIONNXData<*> = KITensor.create(proto)

    companion object {
        suspend operator fun invoke(proto: GraphProto, opSetRegistry: OperatorSetRegistry): KIGraph {
            val valueOrderInfo = GraphValueOrderInfo()
            val nodes = proto.collectOperators<KIONNXData<*>>(valueOrderInfo)
            val operators = ArrayList<Operator<KIONNXData<*>, KIONNXData<*>>>(nodes.size).apply {
                for (node in nodes) {
                    add(KIOperatorFactory.create(node.proto, opSetRegistry))
                }
            }

            val graph = KIGraph(proto, operators, valueOrderInfo)
            for (operator in graph.operators) {
                when(operator.info.type) {
                    "LSTM" -> LSTMContext.appendContext(graph.preparedTensorsContext, graph.initializers as List<KITensor>, operator)
                    "DynamicQuantizeLSTM" -> DynamicQuantizeLSTMContext.appendContext(graph.preparedTensorsContext, graph.initializers as List<KITensor>, operator)
                    "GRU" -> GRUContext.appendContext(graph.preparedTensorsContext, graph.initializers as List<KITensor>, operator)
                    "Attention" -> AttentionContext.appendContext(graph.preparedTensorsContext, graph.initializers as List<KITensor>, operator)
                    "QAttention" -> QAttentionContext.appendContext(graph.preparedTensorsContext, graph.initializers as List<KITensor>, operator)
                    "MatMulInteger" -> MatMulIntegerVer10.MatMulIntegerPrepare.appendContext(graph.preparedTensorsContext, graph.initializers as List<KITensor>, operator)
                }
            }
            return graph
        }
    }
}
