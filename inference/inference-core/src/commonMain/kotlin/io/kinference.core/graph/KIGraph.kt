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
import io.kinference.graph.GraphContext
import io.kinference.graph.Graph
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

@ExperimentalTime
class KIGraph(proto: GraphProto, opSetRegistry: OperatorSetRegistry) : Graph<KIONNXData<*>>(proto, opSetRegistry, KIOperatorFactory) {
    private val preparedTensorsContext = GraphContext<KIONNXData<*>>()

    init {
        initializers as List<KITensor>
        for (operator in operators) {
            when(operator.info.type) {
                "LSTM" -> LSTMContext.appendContext(preparedTensorsContext, initializers as List<KITensor>, operator)
                "DynamicQuantizeLSTM" -> DynamicQuantizeLSTMContext.appendContext(preparedTensorsContext, initializers as List<KITensor>, operator)
                "GRU" -> GRUContext.appendContext(preparedTensorsContext, initializers as List<KITensor>, operator)
                "Attention" -> AttentionContext.appendContext(preparedTensorsContext, initializers as List<KITensor>, operator)
                "QAttention" -> QAttentionContext.appendContext(preparedTensorsContext, initializers as List<KITensor>, operator)
                "MatMulInteger" -> MatMulIntegerVer10.MatMulIntegerPrepare.appendContext(preparedTensorsContext, initializers as List<KITensor>, operator)
            }
        }
    }

    override fun makeContext(root: GraphContext<KIONNXData<*>>?): GraphContext<KIONNXData<*>> {
        val context = GraphContext(root)
        context.mergeContext(preparedTensorsContext)
        return context
    }

    override fun prepareInput(proto: TensorProto): KIONNXData<*> = KITensor.create(proto)
}
