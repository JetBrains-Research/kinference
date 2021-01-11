package io.kinference.algorithms.gec.preprocessing

/**
 * abstract class-constant for Tag
 */
abstract class Tag {
    abstract val value: String
}

/**
 * Tag which represents KEEP tag
 */
object TagKeep : Tag() {
    override val value: String = "\$KEEP"
}

/**
 * Tag which represents PAD tag
 */
object TagPad : Tag() {
    override val value: String = "@@PADDING@@"
}

/**
 * Tag which represents UNKNOWN tag
 */
object TagUnk : Tag() {
    override val value: String = "@@UNKNOWN@@"
}

/**
 * Tag which represents CORRECT tag
 */
object TagCorrect : Tag() {
    override val value: String = "CORRECT"
}

/**
 * Tag which represents INCORRECT tag
 */
object TagIncorrect : Tag() {
    override val value: String = "INCORRECT"
}

/**
 * Tag which represents DELETE tag
 */
object TagDelete : Tag() {
    override val value: String = "\$DELETE"
}
