package io.kinference.algorithms.gec.preprocessing

abstract class Tag{
    abstract val value: String
}

object tagKeep: Tag() {
    override val value: String = "\$KEEP"
}

object tagPad: Tag() {
    override val value: String = "@@PADDING@@"
}

object tagUnk: Tag() {
    override val value: String = "@@UNKNOWN@@"
}

object tagCorrect: Tag() {
    override val value: String = "CORRECT"
}

object tagIncorrect: Tag() {
    override val value: String = "INCORRECT"
}

object tagDelete: Tag() {
    override val value: String = "\$DELETE"
}
