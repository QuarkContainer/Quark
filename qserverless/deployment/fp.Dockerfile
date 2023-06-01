# syntax=docker/dockerfile:1
   
FROM ubuntu:latest
WORKDIR /app
#COPY ./target/debug/func_pod app
#ADD ./fp_logging_config.yaml /app/fp_logging_config.yaml
COPY . .
CMD ["./func_pod"]
#EXPOSE 3000

