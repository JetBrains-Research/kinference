FROM ubuntu:20.04

RUN apt-get update

ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y wget gnupg xvfb firefox git

RUN wget -q -O - https://apt.corretto.aws/corretto.key | apt-key add - \
    && echo "deb https://apt.corretto.aws stable main" >> /etc/apt/sources.list.d/amazoncoretto.list

RUN apt-get update && apt-get install -y java-17-amazon-corretto-jdk
