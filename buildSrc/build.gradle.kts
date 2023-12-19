repositories {
    mavenCentral()
}

plugins {
    `kotlin-dsl`
}

dependencies {
    implementation("org.jetbrains.kotlin:kotlin-gradle-plugin:1.8.22")
    implementation(gradleApi())
    implementation(gradleKotlinDsl())
    api("com.amazonaws:aws-java-sdk-s3:1.11.896")
}
