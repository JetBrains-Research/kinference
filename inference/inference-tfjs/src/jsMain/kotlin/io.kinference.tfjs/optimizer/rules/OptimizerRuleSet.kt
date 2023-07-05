package io.kinference.tfjs.optimizer.rules

import io.kinference.tfjs.optimizer.rules.context.GRUContextRule
import io.kinference.tfjs.optimizer.rules.context.LSTMContextRule

object OptimizerRuleSet {
    val DEFAULT_OPT_RULES = listOf(
        GRUContextRule,
        LSTMContextRule
    )
}
