package io.kinference.algorithms.gec.preprocessing

abstract class Tag {
    abstract val value: String
}

object TagKeep : Tag() {
    override val value: String = "\$KEEP"
}

object TagPad : Tag() {
    override val value: String = "@@PADDING@@"
}

object TagUnk : Tag() {
    override val value: String = "@@UNKNOWN@@"
}

object TagCorrect : Tag() {
    override val value: String = "CORRECT"
}

object TagIncorrect : Tag() {
    override val value: String = "INCORRECT"
}

object TagDelete : Tag() {
    override val value: String = "\$DELETE"
}
