package io.kinference.tfjs.graph

import io.kinference.graph.GraphContext
import io.kinference.graph.Graph
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.operators.TFJSOperatorFactory

class TFJSGraph(proto: GraphProto, opSetRegistry: OperatorSetRegistry) : Graph<TFJSData<*>>(proto, opSetRegistry, TFJSOperatorFactory) {
    override fun prepareInput(proto: TensorProto): TFJSData<*> = TFJSTensor.create(proto)
    override fun makeContext(root: GraphContext<TFJSData<*>>?): GraphContext<TFJSData<*>> = TFJSGraphContext(root as? TFJSGraphContext)
    override fun cleanupUntilOrder(context: GraphContext<TFJSData<*>>, order: Int) {
        context.removeValues { it !in availableInputs && valueOrderInfo.getOrder(it) <= order }
    }
}
