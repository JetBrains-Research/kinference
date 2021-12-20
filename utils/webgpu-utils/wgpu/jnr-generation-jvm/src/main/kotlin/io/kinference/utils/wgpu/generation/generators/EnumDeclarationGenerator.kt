package io.kinference.utils.wgpu.generation.generators

import com.squareup.kotlinpoet.*
import io.kinference.utils.wgpu.generation.generators.base.EnumClassGenerator
import io.kinference.utils.wgpu.generation.models.EnumDeclaration

fun generateEnumDeclaration(
    packageName: String,
    enumDeclaration: EnumDeclaration,
): TypeSpec = EnumDeclarationGenerator(packageName, enumDeclaration).generate()

class EnumDeclarationGenerator(
    packageName: String,
    private val enumDeclaration: EnumDeclaration
) : EnumClassGenerator(ClassName(packageName, enumDeclaration.name)) {
    override fun generateImpl() {
        builder.apply {
            val valueName = "value"
            primaryConstructor(
                FunSpec.constructorBuilder()
                    .addParameter(valueName, Int::class)
                    .build()
            )
            addProperty(
                PropertySpec.builder(valueName, Int::class)
                    .addAnnotation(JvmField::class)
                    .initializer(valueName)
                    .build()
            )
            enumDeclaration.options.forEach { option ->
                addEnumConstant(
                    enumOptionName(option.name),
                    TypeSpec.anonymousClassBuilder()
                        .addSuperclassConstructorParameter("%L", option.value)
                        .build()
                )
            }
        }
    }

    private fun enumOptionName(cName: String) =
        cName
            .removePrefix("${enumDeclaration.name}_")
            .removePrefix("${enumDeclaration.name.removeSubstring("Native")}_")
            .let { name ->
                if (name.first().isDigit()) {
                    "E$name"
                } else {
                    name
                }
            }

    private fun String.removeSubstring(substring: String): String {
        val index = indexOf(substring)
        return if (index == -1) {
            this
        } else {
            removeRange(index, index + substring.length)
        }
    }
}
