package io.kinference.utils.wgpu.generation.utils

import org.antlr.v4.runtime.ParserRuleContext
import org.antlr.v4.runtime.tree.ParseTree
import org.antlr.v4.runtime.tree.TerminalNode

fun ParseTree.terminalNodes(): List<TerminalNode> =
    when (this) {
        is ParserRuleContext -> children.flatMap { it.terminalNodes() }
        is TerminalNode -> listOf(this)
        else -> error("Unknown node type")
    }
