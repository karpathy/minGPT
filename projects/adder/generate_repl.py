import os
import re
import json
import torch
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN

# ensure adder training is run
if not os.path.isfile("out/adder/config.json") or not os.path.exists("out/adder/model.pt"):
    print ("please run adder.py")
    exit()

# load config
C = CN()
config = json.loads(open("out/adder/config.json").read())
C.merge_from_dict(config["model"])
if C.vocab_size is None:
    C.vocab_size = 10
if C.block_size is None:
    C.block_size = config["data"]["ndigit"] * 3

# load model checkpoint
model = GPT(C)
checkpoint = torch.load('out/adder/model.pt')
model.load_state_dict(checkpoint)

# ask
# 99 + 99
# 10 + 20
# 10 + 23
input_expr = "99 + 99"
print (">>>", input_expr)
try:
    while True:
        input_digits = re.findall(r"(\d){1}", re.sub(r"[^0-9]", "", input_expr))
        max_result_size = len(input_digits)-1 # (99 + 99) {4} = 198 {3} -> max_result_size = len-1
        result = model.generate(torch.tensor([[int(d) for d in input_digits]]), max_new_tokens=max_result_size)
        output_result = result[0][len(input_digits):].flip(dims=(0,)).tolist() # output was trained in flipped manner.

        print ("".join([str(d) for d in output_result]).lstrip("0"))
        input_expr = input(">>> ").strip()

        if input_expr.startswith("exit") or input_expr.startswith("quit"):
            exit()
except KeyboardInterrupt:
    exit()
