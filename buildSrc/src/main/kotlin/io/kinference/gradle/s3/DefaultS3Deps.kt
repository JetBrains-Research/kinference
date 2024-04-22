package io.kinference.gradle.s3

fun S3Dependency.Context.defaultS3Deps() {
    s3Test("bert:standard:en:v1")
    s3Test("bert:gec:en:standard:v2")
    s3Test("gpt2:flcc-py-completion:quantized:v3")
    s3Test("gpt2:flcc-py-completion:standard:v3")
    s3Test("gpt2:grazie:distilled:quantized:v6")
    s3Test("gpt2:r-completion:standard:v1")
    s3Test("gpt2:r-completion:quantized:v1")
    s3Test("catboost:ij-completion-ranker:v1")
    s3Test("catboost:license-detector:v1")
    s3Test("custom:comment_updater")
    s3Test("bert:en_tree:quantized")
    s3Test("bert:electra")
}
