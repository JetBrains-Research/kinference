FROM ubuntu:22.04

ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get install -y wget tar bzip2 xvfb git gnupg

# Install firefox
RUN wget -O ~/Firefox.tar.bz2 "https://download.mozilla.org/?product=firefox-latest&os=linux64" && \
    tar xjf ~/Firefox.tar.bz2 -C /opt/ && \
    ln -s /opt/firefox/firefox /usr/bin/ && \
    rm ~/Firefox.tar.bz2

#Install Amazon Corretto JDK 17
RUN wget -q -O - https://apt.corretto.aws/corretto.key | apt-key add - \
    && echo "deb https://apt.corretto.aws stable main" >> /etc/apt/sources.list.d/amazoncoretto.list

RUN apt-get update && \
    apt-get install -y java-17-amazon-corretto-jdk
