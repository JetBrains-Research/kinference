package io.kinference.tfjs.graph

import io.kinference.graph.Context
import io.kinference.graph.Graph
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.operators.TFJSOperatorFactory

class TFJSGraph(proto: GraphProto, opSetRegistry: OperatorSetRegistry) : Graph<TFJSData<*>>(proto, opSetRegistry, TFJSOperatorFactory) {
    override fun prepareInput(proto: TensorProto): TFJSData<*> = TFJSTensor.create(proto)
    override fun makeContext(root: Context<TFJSData<*>>?): Context<TFJSData<*>> = TFJSContext(root as? TFJSContext)
}
