#! /bin/bash
#rm -f /var/log/quark/pytorch*.log

serve_name="pytorch_llama_serve_gpu"

namespace_def="curl --header \"Content-Type: application/json\" --request POST --data '{\"tenant\":\"t1\", \"namespace\":\"ns1\", \"revision\": 0, \"disable\": false}' http://localhost:4000/namespaces/"

#func_del="curl --request \"DELETE\" http://localhost:4000/funcpackages/t1/ns1/$serve_name"

func_def="curl --header \"Content-Type: application/json\" --data '{ \"tenant\":\"t1\", \"namespace\":\"ns1\", \"funcname\": \"$serve_name\", \"revision\": 0, \"image\": \"llama\", \"commands\": [\"python3\", \"/cchen/service_scricpt/pytorch_llama1b.py\"], \"envs\": [[\"LD_PRELOAD\", \"/cchen/Quark/target/release/libcudaproxy.so\"], [\"HF_DATASETS_OFFLINE\",\"1\"], [\"TRANSFORMERS_OFFLINE\",\"1\"], [\"LD_LIBRARY_PATH\", \"/cchen/Quark/target/release/:$LD_LIBRARY_PATH\"]] }' -X POST  http://localhost:4000/funcpackages/ | jq";

func_poll="curl http://localhost:4000/funcpackages/t1/ns1/$serve_name"

func_call="time curl -v -H \"Content-Type: application/json\" --data '{ \"tenant\":\"t1\", \"namespace\":\"ns1\", \"funcname\": \"$serve_name\", \"prompt\": \"What is your name\" }' -X POST http://localhost:4000/funccall/ | jq"


eval "$namespace_def"
#eval "$func_pool"
if eval "$func_poll" | grep 'NotExist'; then
   printf "\nFunction not defined. Defining...\n"
   eval "$func_def"
fi

printf "\n Calling Function...\n"
# ret=$("eval $func_call")
eval "$func_call"
